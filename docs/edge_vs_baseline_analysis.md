**在“没升云”的那几段，其实并不是“和 edge-only baseline 完全一样地在生成”**。两套流程对 edge 侧的「下一块 token 的起始条件」与「反重复约束」并不相同，这正是第一次出现重复 token 的根因。

下面我把两者的差别、导致第一次重复的直接机制、以及可操作的修改点讲清楚。

---

# 发生了什么（差别点）

## 1) baseline 的生成方式
edge-only baseline 走的是 HuggingFace 的 `generate` 流程，**每一步都拿“完整的上下文”去采样**，而且默认是**启用采样**（`do_sample=True`，配合 `temperature/top_p`），并带有停止准则/去重约束（具体在你的 `StreamingGenerator`/generate 调用链里）。这可以天然地打破局部循环。你在 edge 侧的实现里就能看到 `generate(... do_sample=True, temperature=..., top_p=...)` 的调用：

> `self.model.generate(..., temperature=temperature, top_p=top_p, do_sample=True, ...)`（同文件中后面 block 级的生成循环也沿用这一套）

## 2) speculative decoding 中 edge 的生成方式
在 speculative 流程里，你为了能做「K 个草稿 token / 增量验收」，改成了**基于 KV cache 的逐 token 增量**：

- 先 **prefill** 一次，把多模态输入喂给 `thinker` 拿到 `past_key_values`，并把**“最后一个输入 token”缓存到 `last_token_id`**：
- 然后每个草稿块调用 `_generate_draft_tokens_incremental(...)`，**从 `last_token_id` 出发**，只喂 `[1,1]` 的 `input_ids` + `past_key_values` 做 **贪心/轻采样**地扩展 K 个 token：
- 草稿通过或纠错后，再把**被接受的 token**追加回 `context`，同时**尝试推进 KV**（`_advance_kv_cache`）：

这套“增量式 + 只喂最后一个 token”的策略本身没问题，但**你的 `last_token_id` 维护有漏洞** + **反重复约束远弱于 baseline**，两者叠加就会触发「低不确定度但持续重复」的现象。

---

# 第一个重复 token 为什么会出现（根因）

**根因 A：`last_token_id` 没有在“接受草稿后”同步更新**  
- 在 `_generate_draft_tokens_incremental(...)` 里，起点是 `context['last_token_id']`；如果有就从它开始，没有才退回到 `input_ids` 的最后一位：
- 但本函数结尾**只把 `past_key_values` 与 `last_generated_tokens` 写回 `context`**，**没有把新的“最后一个 token”回写到 `last_token_id`**（下一块草稿仍然会从旧的 `last_token_id` 起跳）。
- 你在纠错时有专门 **回滚 `last_token_id`** 的逻辑（防止从错误位置继续），但**在“全通过”或“低不确定度直接接受”路径中并没有**把 `last_token_id` 更新到“刚刚接受的最后一枚 token”：

*效果*：  
进入下一轮草稿生成时，**KV 已被推进**（`_update_context_incremental` 会把已接受 token 推进 KV），但**step=0 的 seed 仍是旧的 `last_token_id`**。这会造成一种「teacher-forcing 位置错位」：用**同一个起始 token** + **几乎相同的 KV 状态**去再次预测下一步，模型会**稳定地给出同一个后续 token**。如果这个 token 的置信度很高（低熵），你的阈值不会触发升云，于是就出现了你日志里那种  
`78685, 78685, 78685, …` 的块内/块间重复。

**根因 B：增量解码时的反重复保护远弱于 baseline**  
- baseline 里（`generate` 路径）通常会启用 **`no_repeat_ngram_size`/`repetition_penalty`**、再加上**采样**，这些能在序列尾部避免一遍遍地“吃”同一个 token（就算模型很自信）。  
- 你的增量实现里只有一个**很轻的“最近 4 个里出现 ≥2 次就 top-k 采样”**的小补丁，而且只看**当前草稿块内部**：
  并**没有**像 baseline 那样对**跨块的 n-gram**或**全局频次**做抑制，也没有显式的 `repetition_penalty` 处理。

*效果*：  
一旦 A 导致每块草稿都从**同一 seed**起跳，B 又没有足够强的去重，循环就会被“自信地”延续；而你又是**以不确定度作为升云触发**，这种“自信重复”反而满足“低不确定度 → 不升云”的条件，于是就“越陷越深”。

---

# 怎么改（最小改动 → 稳定修复）

我给出两步**最小侵入式修复**，基本不动你的整体框架：

### ✅ 修复 1：不要依赖 `last_token_id`；始终从 **`context['input_ids'][:, -1:]`** 起步
直接把 `_generate_draft_tokens_incremental(...)` 的起始逻辑改成**总是**用已更新的 `input_ids` 尾部作为 step=0 的种子；这样**只要 `_update_context_incremental` 已把接受的 token 追加到 `input_ids`**，下一块就一定从“真正的最新 token”起步，不会被陈旧的 `last_token_id` 卡住。

把这段（精简后）：
```python
if 'last_token_id' in context and context['last_token_id'] is not None:
    current_input_ids = context['last_token_id']  # [1,1]
else:
    current_input_ids = context['input_ids'][0, -1:].unsqueeze(0)
```
替换为：
```python
current_input_ids = context['input_ids'][:, -1:].contiguous()
```
（对应位置见增量函数起始处的分支判断）

> 这样你甚至可以**完全移除 `last_token_id` 这块状态**；在回滚时，你已经把 `input_ids` 回退到正确长度（你在纠错路径里对 `input_ids` 的回滚逻辑是有的），因此**从 `input_ids` 尾部起跳足够**。

如果你暂时不想移除 `last_token_id`，也至少要在**每次接受 token 后**把它更新为最后一枚：
- 在 `_update_context_incremental(...)` 里，追加完 token 后加：
  ```python
  last_id = new_context['input_ids'][0, -1:].clone().unsqueeze(0)
  new_context['last_token_id'] = last_id
  ```
  （该函数正是你接受 token 后统一更新上下文的地方）
- 同时，在 `_generate_draft_tokens_incremental(...)` 的函数尾，也把 **本轮生成的最后一枚**写回去（双保险）

### ✅ 修复 2：把 baseline 的反重复思路补到增量解码
在 `_generate_draft_tokens_incremental(...)` 的**每步**得到 `logits` 后，按最近若干 token（建议把 `context['input_ids']` 的末尾 64～128 token 与 `draft_tokens` 合并考虑）做一个**简易 `repetition_penalty`** 与 **`no_repeat_ngram_size`** 约束。示例（伪代码）：

```python
# recent window
recent = torch.cat([context['input_ids'][0, -128:], torch.tensor(draft_tokens, device=...)])

# repetition penalty（例如 1.05~1.2）
penalty = 1.1
unique_ids = torch.unique(recent)
logits[unique_ids] /= penalty

# no_repeat_ngram_size（例如 2 或 3）
def block_ngram(logits, history, n=3):
    if history.shape[0] < n-1: 
        return
    # 找出所有以 history 结尾的 (n-1)-gram，阻止继续补成某个 n-gram
    # （可用哈希或字典加速；也可以只对“最后 n-1 个”精确阻断）
    # 对应 token 的 logits -= 1e9

block_ngram(logits, torch.cat([context['input_ids'][0], torch.tensor(draft_tokens, device=...)]), n=3)
```

你已经有一个“最近 4 个里出现 ≥2 次就 top-k 采样”的轻量保护：fileciteturn26file1  
把它**升级**为上述两个约束（哪怕只做 `repetition_penalty` 也会立刻改善）。

---

# 额外建议（可选但推荐）

- **统一使用 `thinker`** 做 prefill 和增量（你现在就是这么做的，挺好），但注意 `_advance_kv_cache(...)` 里也要用同一个子模型推进 KV（你现在已经是用 `thinker` 推 KV 的）：
- **跨块的循环检测**：你目前的循环探测只看“当前草稿块的 `draft_tokens`”。可以把 `context['input_ids']` 尾部的若干 token 一起纳入检测，以捕获“块与块之间的重复模式”。  
- **停止准则**：baseline 里用的是句子结束符/最小长度类的 stopping criteria；speculative 这边你也已经有 token 级停止器（`StopWhenNSentences`）可用，确保在低温/低熵时不会无限“磨”同一个 token。

---

# 一句话总结

- **第一次重复 token 的直接诱因**：增量生成的 **seed token** 没跟着“已接受 token”更新，下一块草稿又从**同一个旧 seed**起跳；而增量路径缺少像 baseline 那样的**强力反重复约束**，于是模型在“低不确定度”的情况下自信地重复，**不会触发升云**。  
- **修复思路**：**总是从 `input_ids` 的最后一个 token 起步**（或确保 `last_token_id` 和它同步）；并**把 repetition penalty / no-repeat-ngrams** 引入增量解码。

如果你愿意，我可以直接把上述两个改动补成 patch 版本（针对 `speculative_decoding.py` 的具体函数段落）。另外，baseline 的 edge-only 实现调用 `generate(... do_sample=True, temperature/top_p ...)` 这一点也解释了它几乎不出现“自信重复”的现象（它会被采样与去重约束打断）。
