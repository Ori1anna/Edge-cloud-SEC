你这版的 `speculative_decoding.py` 已经把我之前提的几件大事做上了（prefill 拿 KV、从 `input_ids[-1]` 做增量、纠错时回滚 `input_ids` 和 KV、把音频只在首步带入等）。但日志里仍然会出现 “ABAB…” 这种环形重复，根因主要有三类，其中**第一条是决定性 bug**：

---

## 结论直说（最关键的两点）

1) **KV 被“重复前推”（double-advance）**  
   - 你在 `_generate_draft_tokens_incremental` 里生成 k 个草稿 token 后，已经把 `context['past_key_values']` 更新到了这 k 个 token 之后。  
   - 随后无论是“全接收”还是“纠错后接收”，你又调用 `_update_context_incremental(...)`，而这个函数里**再次**用这些 token 去前推 KV。  
   - 结果：KV 内部状态比 `input_ids` **超前了整整一块**，下一轮你又从 `input_ids[-1]` 喂一次，这个 token 实际在 KV 里已经“看过”，模型就非常容易把刚才那段又走一遍，形成 ABAB 循环（你的日志里像 `[104432, 26381, 104432, 26381, ...]` 这一类就是典型表现）。

   **修法（最小改动）**  
   - 让 `_update_context_incremental` **只更新 `input_ids`/`attention_mask`，不要再推进 KV**。  
   - 即：把 `_update_context_incremental` 中对 `_advance_kv_cache(...)` 的调用删掉/加开关默认关闭；保持 KV 的推进只发生在两处：  
     1）草稿增量生成函数内部（你现在已做）；  
     2）发生纠错回滚后，用“被接受的 token 序列”手动推进一次（你现在也已做），**但这时再调用 `_update_context_incremental` 时也不要再推进**。

2) **重复惩罚与 no-repeat 约束实现有问题，反而会“喂高”重复 token 概率**  
   - 你把“重复惩罚”写成了“对最近出现过的 token 统一 `logits /= 1.1`”。对**负值 logit** 这么做会让它**变得不那么负**→ 概率**上升**，跟 HF 的做法相反（HF 是：`score>0` 才除，`score<=0` 要乘）。  
   - 你标注为 “no_repeat_ngram_size=2” 的实现，实际上只是**禁止 “A A” 紧邻重复**：你是把“prefix=最近 1 个 token 本身”直接打成 `-1e6`，这根本**不是**“不重复 2-gram”（应该是：当上下文最后 1 个 token 为 `x` 时，**禁止所有历史中出现过的 `(x, Y)` 的 `Y`**）。现在像 “A B A B …” 的 2-gram 循环完全不会被你这段代码阻断。

   **修法（最小改动）**  
   - **HF 风格重复惩罚**（示例）：
     ```python
     # recent_tokens = 历史 + 当前草稿 最近窗口
     rp = 1.1  # 或 1.2
     for t in set(recent_tokens):
         if logits_temp[t] > 0:
             logits_temp[t] /= rp
         else:
             logits_temp[t] *= rp
     ```
   - **真正的 no-repeat 2-gram**（基于历史构建 banned next tokens）：
     ```python
     def build_bigrams_ban(history: list[int]) -> dict[int, set[int]]:
         banned = {}
         for a, b in zip(history[:-1], history[1:]):
             banned.setdefault(a, set()).add(b)
         return banned

     history = context['input_ids'][0].tolist() + draft_tokens
     banned_map = build_bigrams_ban(history)
     prev = history[-1] if history else None
     if prev in banned_map:
         logits_scaled[list(banned_map[prev])] = -float('inf')
     ```
   - 如此才会把 “AB 循环” 在 `B` 那步直接封死。

---

## 你这版里做对了的 & 还需微调的

- ✅ **prefill 拿 KV** 并存 `last_token_id`，增量从 `input_ids[-1]` 开始，这是对的。  
- ✅ **纠错路径**里有“回滚 `input_ids`+KV 到草稿前，再用已接受 token 推进 KV”的思路，这步思路正确；但是**随后的 `_update_context_incremental` 又把 KV 推进了一遍**，导致上面的 double-advance 问题再次出现。  
- ⚠️ **采样策略**：你把增量解码改成 `temperature=0.7 + top_p=0.9` 的采样（而非 baseline 的 `generate(..., do_sample=False, no_repeat_ngram_size=2, repetition_penalty=...)`）。只要门槛（entropy 阈值）较高，很多“略不确定”的 token 会被**直接接受**，这会明显放大上面两条（KV 失配 + 约束失效）带来的重复风险。  
- ⚠️ **entropy 阈值设为 4.0** 明显偏高（你日志里 3.x 的不确定度一律放过）。在约束没完全正确前，阈值过高会让“坏块”不断被边缘模型自循环地产生并被接受。  

---

## 精准改动清单（最小侵入）

> 下面的代码块都是“替换/删除”级别的小改，保证一次只推进一次 KV，且把重复约束修正为有效版本。你可以逐条改。

### 1) `_update_context_incremental`：移除/默认关闭“推进 KV”的逻辑
把里面这段“推进 KV”的代码删掉或加个默认 False 的开关（建议直接删掉）：
```python
# ❌ 删掉这整段
if 'past_key_values' in new_context and new_context['past_key_values'] is not None:
    try:
        new_context['past_key_values'] = self._advance_kv_cache(
            new_context['past_key_values'],
            new_tokens,
            self.edge_model.device
        )
    except Exception as e:
        ...
```
只保留**拼接 `input_ids`/`attention_mask`**与更新 `last_token_id`。  
> 这样，“KV 前推”严格只发生在：  
> a) 草稿增量函数内部（正常快进）；  
> b) **纠错回滚**后你**手工**用“已接受 token”推进一次。

### 2) 纠错路径里 **不要** 再次推进 KV
你现在纠错分支是：回滚 → `_advance_kv_cache(kv_before_draft, accepted+correction)` → `_update_context_incremental(...)`。  
按第 1 条改完后，**这一步就不会再次前推**，状态一致。

### 3) 修正重复惩罚与 2-gram 约束（在 `_generate_draft_tokens_incremental` 里）
把你的“统一 `/= 1.1`”替换成 HF 风格的正负号分支；把“只打上一个同前 token 的 id”改成真正的 **2-gram ban**。示例（放在采样前）：
```python
# 重复惩罚（HF 风格）
recent_window_size = 64
recent_tokens = []
if context['input_ids'].shape[1] > 0:
    context_recent = context['input_ids'][0, -min(recent_window_size, context['input_ids'].shape[1]):]
    recent_tokens.extend(context_recent.tolist())
recent_tokens.extend(draft_tokens)

rp = 1.1  # 你可微调
for t in set(recent_tokens):
    if logits_temp[t] > 0:
        logits_temp[t] /= rp
    else:
        logits_temp[t] *= rp

# 真·no-repeat 2-gram
def build_bigrams_ban(history):
    banned = {}
    for a, b in zip(history[:-1], history[1:]):
        banned.setdefault(a, set()).add(b)
    return banned

history = (context['input_ids'][0].tolist() + draft_tokens)
if history:
    banned_map = build_bigrams_ban(history)
    prev = history[-1]
    if prev in banned_map and banned_map[prev]:
        logits_scaled[list(banned_map[prev])] = -float('inf')
```

### 4) （可选但强烈建议）把“循环检测”升级为“触发一次性升云校验”，而不是直接 `break`
你现在检测到 cross-block pattern 后直接 `break`。更稳妥的做法是在该批次**强制 `needs_cloud_verification=True`**，用云端概率把这个循环切断，然后继续。  

---

## 为什么 baseline 不会重、而你的会重？

- **baseline** 很可能是一次性 `model.generate(...)`，内部 KV 与 `input_ids` 始终由 HF 框架自己维护，并且有**正确**的 `no_repeat_ngram_size` + `repetition_penalty` 实现（且多用贪心或更保守的采样）。  
- 你的 **spec-dec 增量**路径里：  
  1) 出现了 **KV 双推进** → **上下文错位**；  
  2) 约束实现不正确 → **AB 循环不被禁止**；  
  3) 采样 + 高阈值 → **很多“边缘不确定”的 token 被直接接受**，让 1)+2) 的影响被放大。

把上面四条补丁打上去后，和 baseline 的行为就会明显接近：  
- 不会再在“低不确定”分支下被 Edge **自我循环**；  
- 真正的 **2-gram** 与正确的 **重复惩罚**会把 “地方的地方的 / 104432,26381,…” 这类循环直接压住；  
- 纠错回滚后的推进与 `input_ids` 完全对齐，不会再出现“KV 已经在前、`input_ids` 还在后”的错位。

—

如果你愿意，我也可以把这几处改动直接整理成一个小 patch（diff）再给你。  
（以上分析与修改点均基于你贴出的 `speculative_decoding.py` 当前内容。fileciteturn27file0）
