# Speculative Decoding（中文标点“逗号泛滥/被截断”）修复与优化指南

> **目标**：在不改变整体框架的前提下，修正 `_generate_draft_tokens_incremental()` 中因中文分词粒度与多重惩罚叠加导致的 **逗号/顿号滥用** 与 **早停/截断感**，并给出最小侵入的可复制代码段。

---

## TL;DR（先说结论）

- **根因 1：中文 + `no_repeat_ngram_size=2` 等价逻辑**  
  你在增量草稿里实现了“真·2-gram 禁复用”（bigram ban）。中文大多是**单字级 token**，2-gram 禁令会把大量“合理的下一字”打成 `-inf`，模型为绕过禁令**最容易选标点**（逗号/顿号/冒号）。

- **根因 2：对内容 token 的抑制 > 对标点的抑制**  
  同时叠加了 `repetition_penalty`（乘法）和 `presence_penalty`（线性减分），而对标点只在“**滞后触发**”时才减分。结合根因 1，**逗号仍是相对“高分”**，于是“字，字，字……”出现。

- **根因 3：标点抑制滞后 & 力度不足**  
  目前“最近 6 个草稿位里 ≥3 个标点才 -5 分”，当模型已进入“逐字 + 逗号”后，**来得太晚**。

**解决思路**：  
1) 中文**取消 2-gram 禁令**（或升到 ≥3-gram 且仅对非中文）；  
2) 加入**前瞻式硬闸门**（在选 token 前阻断逗号/顿号等），让“内容字的优先级”回到正确轨道；  
3) **弱化/移除 presence_penalty**，留轻微的 repetition_penalty 即可；  
4) 选中被闸的标点时，从 **top-k** 里改选第一个**非标点**；  
5) （可选）删除“滞后触发”的标点密度惩罚，或调得更严格；  
6) **Cloud 校验**侧保持一致的约束，避免边云分歧。

---

## 改动位置总览（最小侵入）

文件：`speculative_decoding.py`  
函数：`_generate_draft_tokens_incremental()` 内 **“选 token 前”** 与 **“2-gram 禁令”** 两处。

---

## 1) 中文禁用 2-gram 禁令 / 非中文使用温和 3-gram

> 将你当前的“真·2-gram 禁复用”替换为**语言感知**版本：**中文不施加 n-gram 禁令**；其它（如英文）用 **3-gram**，且只对最近窗口生效。

**替换代码段：**

```python
# ==== Language-aware n-gram constraint ====
def _is_cjk(token_id):
    s = self.edge_model.processor.tokenizer.decode([token_id], skip_special_tokens=True)
    return any('\u4e00' <= ch <= '\u9fff' for ch in s)  # 粗略判断是否含中文

full_history = context['input_ids'][0].tolist() + draft_tokens

# 若最近窗口出现中文，则不施加 n-gram 禁令；否则对非中文用温和 3-gram
if len(full_history) >= 3 and not any(_is_cjk(t) for t in full_history[-12:]):
    trigrams = {}
    for a, b, c in zip(full_history[:-2], full_history[1:-1], full_history[2:]):
        trigrams.setdefault((a, b), set()).add(c)
    a, b = full_history[-2], full_history[-1]
    banned = list(trigrams.get((a, b), []))
    if banned:
        logits_temp[banned] = -float('inf')
# ==== end n-gram constraint ====
```

> **原因**：中文 token 粒度 ≈ 单字，2-gram 禁令会过度抑制自然续写；英文等（wordpiece）用 3-gram 更温和。

---

## 2) 前瞻式“标点硬闸门”（在选 token 前拦截）

> 在温度缩放与取 `argmax` 之前，加入 **hard gate**：如果距离上一次标点**太近**，直接把逗号/顿号/冒号 **ban 掉**（或强力抑制），句号更严格一些。

**新增代码段（建议放在温度缩放后、argmax 前）：**

```python
# ===== Hard punctuation gate (pre-selection) =====
def _ids_for(chars):
    ids = []
    for ch in chars:
        enc = self.edge_model.processor.tokenizer.encode(ch, add_special_tokens=False)
        if enc:
            ids.append(enc[0])
    return set(ids)

PUNCT_IDS  = _ids_for(['，','。','、','：',':','；','！','？'])
COMMA_LIKE = _ids_for(['，','、','：',':'])   # 容易被滥用的一类
PERIOD_LIKE= _ids_for(['。'])

# 统计“自上次标点以来，连续的中文内容 token 数”
hist = context['input_ids'][0].tolist() + draft_tokens
since_punct = 0
for t in reversed(hist):
    if t in PUNCT_IDS:
        break
    s = self.edge_model.processor.tokenizer.decode([t], skip_special_tokens=True)
    if any('\u4e00' <= ch <= '\u9fff' for ch in s):
        since_punct += 1

# 逗号/顿号/冒号：至少隔 2 个中文内容 token 才允许
if since_punct < 2:
    logits_temp[list(COMMA_LIKE)] = -float('inf')

# 句号：至少隔 6 个中文内容 token；如不 ban，则强抑制
if since_punct < 6:
    idx = list(PERIOD_LIKE)
    logits_temp[idx] = torch.where(
        torch.isinf(logits_temp[idx]),
        logits_temp[idx],
        logits_temp[idx] - 8.0
    )
# ===== end gate =====
```

> **原因**：把“插逗号解禁”的捷径**提前**切断，迫使模型继续产出“内容字”，直到真正需要停顿时再给标点。

---

## 3) 弱化/移除 presence_penalty（保留轻微 repetition_penalty）

> 你当前是“乘法式 repetition_penalty + 线性 presence_penalty”叠加，容易把**合理高频字**也压下去，让标点反成为相对高分。

**建议**：
- 先 **移除 presence_penalty** 观察效果；若担心复读，再设到极小值（如 `0.02`），且**只对非标点 token 应用**。  
- 保留 `repetition_penalty` 于 `1.1 ~ 1.2`。

**示例修改：**

```python
# presence_penalty 建议先移除；若必须保留：
presence_penalty = 0.02  # 极小
from collections import Counter
token_counts = Counter([t for t in recent_tokens if t not in PUNCT_IDS])
for token_id, count in token_counts.items():
    logits_temp[token_id] -= (presence_penalty * count)
```

---

## 4) 兜底：若 argmax 命中被闸的标点，从 top-k 里挑第一个“非标点”

> 在 `next_token = argmax(...)` 之后添加：

```python
# 若命中硬闸，则在 top-k 内改选非标点
next_token = torch.argmax(logits_scaled, dim=-1).item()
if next_token in PUNCT_IDS and since_punct < 2:
    top_k = 8
    topk_logits, topk_idx = torch.topk(logits_scaled, top_k)
    for idx in topk_idx:
        if idx.item() not in PUNCT_IDS:
            next_token = idx.item()
            break
```

> **原因**：保持整体 **确定性（do_sample=False）**，仅在违规时优雅降级，避免再次走回“逗号捷径”。

---

## 5)（可选）删除“滞后触发”的标点密度惩罚

你现在还有一段“最近 6 个草稿位里 ≥3 个标点 ⇒ -5 分”的逻辑。加入**硬闸门**后，它容易与其它惩罚叠加出“**标点找替身**（冒号/叹号/语气词）”。建议：

- **移除**该段，或
- 提高阈值（如：最近 8 个里 ≥4）+ 降低力度（-3）。

---

## 6) Cloud 端对齐（避免边云分歧）

若 `CloudModel.verify_tokens(...)` 在校验时仍用“纯贪心、无闸门”，它会**认可**边端的标点倾向。建议在云端也支持：

- `banned_token_ids` / `logit_bias` 入参；或  
- 直接复用同一套 `PUNCT_IDS + 硬闸门` 规则。

这样能保证 **Edge/Cloud 一致** 的生成偏好，减少“接受—回滚—再犯”的循环。

---

## 7) 自测与回归清单

1. **标点比例**：统计生成文本中 **逗号/顿号** 占 token 比例（目标 \< 5%）。  
2. **句内 token 平均长度**：提升表明“逐字+逗号”被抑制。  
3. **N-gram 重复率**：中文下 bigram 重复率可适度上升（自然语言特性所致），但 **标点间隔**明显增大。  
4. **BLEU / CIDEr / BERTScore**：应较“逗号泛滥”版本回升或持平。  
5. **云端接受率**：标点被硬闸后，云端校验仍可通过，`correction_rate` 不应大幅上升。

---

## 8) 可调参建议（按优先级）

1. **硬闸门间隔**：`since_punct < 2`（逗号）与 `< 6`（句号）；根据任务风格 ±1~2。  
2. **repetition_penalty**：1.1~1.2；过大会抑制正常续写。  
3. **presence_penalty**：建议 0 或 ≤ 0.02；且仅作用于非标点。  
4. **top-k 兜底**：`top_k = 8`，可在 5~10 之间微调。  
5. **英文 3-gram**：若英文文本出现机械重复，可将窗口从 12 提到 24。

---

## FAQ

- **会不会导致完全没有逗号？**  
  不会。闸门只是“**最短间隔**”约束，满足最短内容长度后，标点又会自然出现。

- **为什么不直接对标点施加固定大负偏置（logit bias）？**  
  会让模型“找替身”（冒号/顿号/叹号/语气词），反而怪异。**前瞻式“间隔闸门”**更符合语言节奏。

- **采样（top-p/top-k）是不是更自然？**  
  你的场景更偏“**结构化、单句、可控**”，默认保持确定性（贪心）更稳定；我们只在**违规时**用 top-k 兜底。

---

## 一次性对照清单（你需要做的改动）

- [x] **替换**：中文禁用 2-gram；非中文用温和 3-gram（仅最近窗口）。  
- [x] **新增**：前瞻式硬闸门，限制逗号/顿号/冒号最小间隔，句号更严格。  
- [x] **调整**：去掉或极小化 presence_penalty，并只对非标点生效；保留轻微 repetition_penalty。  
- [x] **兜底**：命中闸门时，从 top-k 中改选首个非标点。  
- [x] **（可选）删除**：滞后触发的标点密度惩罚段。  
- [x] **（推荐）云端对齐**：在 Cloud 校验里应用同一套约束。

---

如需，我可以把上述片段直接合并成一个 **最小 diff** 补丁，贴合你当前文件结构进行就地替换。
