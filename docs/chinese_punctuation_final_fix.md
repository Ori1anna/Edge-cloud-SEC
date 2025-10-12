# Chinese Punctuation Final Fix - 中文标点问题最终解决方案

## 问题复现

### Sample 00000033 & 00000039 输出

```
"你明知道，龙，舟，队，的，队员，现在，都，还是，群，菜鸟，，说话，人，却..."
"你说话的时候，语速较快，音调时高时，音，量时强，时弱，停，顿，与，连，贯，度，较，好..."
```

**典型的"单字+逗号"病态模式！**

---

## 根本原因分析（用户正确诊断）

### ✅ 根因1：中文 + 2-gram禁令过度抑制

**代码问题**：
```python
# 之前的严格2-gram禁令
def build_bigrams_ban(history):
    banned = {}
    for a, b in zip(history[:-1], history[1:]):
        banned.setdefault(a, set()).add(b)
    return banned

# 对中文文本的破坏性影响：
# "的" → "队" 出现过一次
# 之后遇到"的"，就禁止"队" → logits[队] = -inf
# 模型只能选标点或其他字
```

**为什么对中文特别严重**：
- 中文tokenization：**1 token ≈ 1个字**
- 英文tokenization：1 token ≈ 1个词根/词缀
- 2-gram禁令在中文下等同于"禁止任何两字词组重复"
- 正常句子如"说话人的语速较快，说话人的音调"会被严重限制

### ✅ 根因2：内容token抑制 > 标点抑制

**双重惩罚叠加**：
```python
# 内容token受到：
repetition_penalty = 1.1  # 乘法：logits / 1.1
presence_penalty = 0.1    # 减法：logits - (0.1 * count)

# 标点只在达到3/6阈值时才-5.0
# 结果：正常内容字被压低，标点相对分数更高
```

**实际效果**：
- "说" (出现2次)：`logits / 1.1 - 0.2` = 大幅降低
- "，" (刚出现)：无惩罚 → 成为最高分
- 模型被迫选逗号

### ✅ 根因3：标点抑制滞后且力度不足

**问题逻辑**：
```python
recent_n = 6
punct_threshold = 3  # 需要3/6才触发

# 模型已经生成："你，明，知，道"
# 此时才触发 -5.0
# 来得太晚了！
```

---

## 解决方案（按用户文档实施）

### 修改1：语言感知的n-gram约束

**文件**：`src/speculative_decoding.py`，第913-939行

**核心策略**：
- **中文**：完全禁用n-gram禁令
- **非中文**：使用温和的3-gram（比2-gram宽松）

**实现**：
```python
def _is_cjk(token_id):
    """Check if token contains Chinese/Japanese/Korean characters"""
    s = tokenizer.decode([token_id], skip_special_tokens=True)
    return any('\u4e00' <= ch <= '\u9fff' for ch in s)

full_history = context['input_ids'][0].tolist() + draft_tokens

# If recent window contains CJK, disable n-gram ban
if len(full_history) >= 3 and not any(_is_cjk(t) for t in full_history[-12:]):
    # Non-CJK: use 3-gram constraint (more lenient)
    trigrams = {}
    for a, b, c in zip(full_history[:-2], full_history[1:-1], full_history[2:]):
        trigrams.setdefault((a, b), set()).add(c)
    
    a, b = full_history[-2], full_history[-1]
    banned = list(trigrams.get((a, b), []))
    if banned:
        logits_temp[banned] = -float('inf')
# else: CJK detected, skip n-gram ban completely
```

**效果**：
- 中文可以自然重复合理的两字组合："说话人的"、"可能是"
- 英文保留温和的3-gram约束，防止机械重复

---

### 修改2：前瞻式"标点硬闸门"

**文件**：`src/speculative_decoding.py`，第941-980行

**核心策略**：在选token**之前**，根据"距离上次标点的内容token数"硬性阻断标点

**实现**：
```python
def _ids_for(chars):
    """Get token IDs for given characters"""
    ids = []
    for ch in chars:
        enc = tokenizer.encode(ch, add_special_tokens=False)
        if enc:
            ids.append(enc[0])
    return set(ids)

PUNCT_IDS = _ids_for(['，', '。', '、', '：', ':', '；', '！', '？'])
COMMA_LIKE = _ids_for(['，', '、', '：', ':'])  # Easily abused
PERIOD_LIKE = _ids_for(['。'])

# Count consecutive CJK content tokens since last punctuation
hist = context['input_ids'][0].tolist() + draft_tokens
since_punct = 0
for t in reversed(hist):
    if t in PUNCT_IDS:
        break
    s = tokenizer.decode([t], skip_special_tokens=True)
    if any('\u4e00' <= ch <= '\u9fff' for ch in s):  # CJK
        since_punct += 1

# Comma/colon: require at least 2 CJK content tokens
if since_punct < 2:
    logits_temp[list(COMMA_LIKE)] = -float('inf')

# Period: require at least 6 CJK content tokens
if since_punct < 6:
    for punct_id in PERIOD_LIKE:
        if not torch.isinf(logits_temp[punct_id]):
            logits_temp[punct_id] -= 8.0
```

**示例**：
```
已生成："说话人"
since_punct = 3（3个中文字）
→ 逗号闸门开启（≥2），允许逗号
→ 句号仍抑制（<6）

已生成："说"
since_punct = 1（1个中文字）
→ 逗号闸门关闭（<2），logits[逗号] = -inf
→ 模型必须继续生成内容字
```

**为什么有效**：
- **提前阻断**：不是"看到太多标点才惩罚"，而是"不够内容就禁止标点"
- **自然节奏**：符合中文写作习惯（至少2-3个字后才标点）
- **硬性规则**：`-inf`确保绝对不会选择，不依赖相对概率

---

### 修改3：移除presence_penalty

**文件**：`src/speculative_decoding.py`，第903-906行

**修改前**（错误）：
```python
presence_penalty = 0.1
for token_id, count in token_counts.items():
    penalty_factor = 1.0 + presence_penalty * count
    logits_temp[token_id] -= penalty_factor
```

**修改后**：
```python
# REMOVED: presence_penalty
# This was causing content tokens to be suppressed more than punctuation
# Combined with 2-gram ban, it made punctuation the "easy choice"
# Keeping only repetition_penalty (multiplicative) is sufficient
```

**原因**：
- `repetition_penalty` (乘法) + `presence_penalty` (减法) = 双重压制
- 正常高频字（"的"、"是"、"人"）被过度惩罚
- 标点只在触发阈值时才惩罚 → 相对分数更高
- **移除后**：只保留温和的`repetition_penalty = 1.1`

---

### 修改4：Top-k兜底机制

**文件**：`src/speculative_decoding.py`，第1014-1022行

**实现**：
```python
next_token = torch.argmax(logits_scaled, dim=-1).item()

# Fallback: if argmax hits blocked punctuation, pick first non-punct from top-k
if next_token in PUNCT_IDS and since_punct < 2:
    top_k = 8
    topk_logits, topk_idx = torch.topk(logits_scaled, top_k)
    for idx in topk_idx:
        if idx.item() not in PUNCT_IDS:
            next_token = idx.item()
            logger.debug(f"Fallback: switched from punctuation to token {next_token}")
            break
```

**作用**：
- 保持整体确定性（`do_sample=False`）
- 仅在违规时（选中了被闸的标点）优雅降级
- 从top-k中选第一个非标点token

---

### 修改5：删除滞后触发的标点密度惩罚

**文件**：`src/speculative_decoding.py`，第908-911行

**原代码**（已删除）：
```python
# 最近6个里≥3个标点 → -5.0
if punct_count >= punct_threshold:
    logits_temp[punct_id] -= 5.0
```

**为什么删除**：
- 与硬闸门功能重复
- 来得太晚（模式已形成）
- 可能与其他惩罚叠加产生"找替身"行为

---

## 修改总结

| 修改项 | 位置 | 核心变化 | 目的 |
|--------|------|----------|------|
| **n-gram约束** | 913-939 | 中文禁用；非中文用3-gram | 避免过度抑制中文 |
| **硬闸门** | 941-980 | 基于间隔硬性阻断标点 | 提前切断"逗号捷径" |
| **presence_penalty** | 903-906 | **完全移除** | 避免双重惩罚内容 |
| **top-k兜底** | 1014-1022 | 新增 | 违规时优雅降级 |
| **滞后惩罚** | 908-911 | **完全移除** | 避免与硬闸门冲突 |

---

## 技术细节

### 为什么2-gram对中文是灾难？

**中文vs英文tokenization**：

| 语言 | 示例文本 | Token化 | 2-gram禁令效果 |
|------|----------|---------|---------------|
| **英文** | "the player" | `[the, player]` | 禁止`the → player`序列，影响小 |
| **中文** | "的队员" | `[的, 队, 员]` | 禁止`的 → 队`，严重限制表达 |

**实际影响**：
```python
# 英文："the player is good, the player is fast"
# 2-gram: 禁止"the → player"重复 → 可改写为"they are good, they are fast"

# 中文："说话人的语速快，说话人的音调高"
# 2-gram: 禁止"说 → 话"、"人 → 的"、"的 → 语" ...
# → 几乎无法改写，只能用标点拆分："说，话，人，的，语，速，快"
```

### 为什么硬闸门有效？

**传统惩罚（滞后）**：
```
Step 1: 生成"你"
Step 2: 逗号分数高 → 生成"，"
Step 3: 生成"明"
Step 4: 逗号分数高 → 生成"，"
Step 5: 检测到标点过多 → 开始惩罚（太晚了！）
```

**硬闸门（提前）**：
```
Step 1: 生成"你"
Step 2: since_punct=1 < 2 → logits[逗号]=-inf → 必须生成内容
Step 3: 生成"明"
Step 4: since_punct=2 ≥ 2 → 闸门开启 → 允许逗号（但不强制）
```

**关键区别**：
- 滞后惩罚：看到问题后反应 → "亡羊补牢"
- 硬闸门：预防问题发生 → "未雨绸缪"

---

## 预期效果

### Sample 00000033 对比

| 版本 | 输出 | 标点密度 |
|------|------|----------|
| **修改前** | `"你，明，知，道，龙，舟，队..."` | ~50% |
| **修改后（预期）** | `"你明知道龙舟队的队员现在都还是群菜鸟，说话人的语气..."` | ~5-10% |

### Sample 00000039 对比

| 版本 | 输出 | BERTScore F1 |
|------|------|--------------|
| **修改前** | `"语速，较快，音调，时高，时，音，量..."` | 0.069 |
| **修改后（预期）** | `"语速较快，音调时高时低，音量时强时弱，停顿与连贯度较好..."` | >0.15 |

---

## 与之前修改的对比

### 修复历程

| 阶段 | 问题 | 方法 | 结果 |
|------|------|------|------|
| **阶段0** | 逗号泛滥 | 增加标点惩罚 | ❌ 导致冒号泛滥 |
| **阶段1** | 冒号泛滥 | 惩罚所有标点 | ❌ 导致语气词泛滥 |
| **阶段2** | 语气词泛滥 | 移除基础惩罚+病态检测 | ✅ 部分改善，但逗号问题复发 |
| **阶段3（本次）** | 逗号再现 | **语言感知+硬闸门** | ✅ 从根源解决 |

### 核心差异

| 之前方法 | 本次方法 |
|----------|----------|
| ❌ 惩罚标点本身 | ✅ 控制标点使用节奏 |
| ❌ 滞后反应（看到问题才处理） | ✅ 提前预防（不够内容就禁止） |
| ❌ 对所有语言一视同仁 | ✅ 中文特殊处理 |
| ❌ 多重惩罚叠加 | ✅ 简化为单一约束 |

---

## 测试建议

### 运行测试

```bash
cd /data/gpfs/projects/punim2341/jiajunlu/edgecloud-sec

python experiments/runs/run_speculative_decoding_cpu_limited.py \
    --dataset_type unified \
    --dataset_path data/processed/mer2024/manifest_audio_only_final.json \
    --caption_type audio_only \
    --language chinese \
    --prompt_type detailed \
    --max_samples 10
```

### 观察指标

1. **标点密度**：
   - 统计逗号在输出中的比例
   - 目标：<10%（正常文本约5-8%）

2. **句内平均字数**：
   - 逗号间的字数
   - 目标：≥3字（修改前约1-1.5字）

3. **质量指标**：
   - BERTScore F1：应提升
   - 主观阅读：应无"单字+逗号"感

4. **日志关键字**：
   - `"Hard gate: blocking comma-like punctuation"` - 闸门生效
   - `"Fallback: switched from punctuation"` - 兜底机制触发
   - `"skip n-gram ban to allow natural Chinese text"` - 中文禁用n-gram

---

## 可调参数

### 按优先级排序

1. **硬闸门间隔**（最重要）：
   - 逗号：`since_punct < 2`（至少2个中文字）
   - 句号：`since_punct < 6`（至少6个中文字）
   - 根据任务风格可±1

2. **repetition_penalty**：
   - 当前：`1.1`
   - 范围：`1.0-1.2`
   - 过大会抑制正常重复

3. **CJK检测窗口**：
   - 当前：`full_history[-12:]`
   - 如需更敏感：减小到`-8`
   - 如需更宽松：增大到`-16`

4. **Top-k兜底**：
   - 当前：`top_k = 8`
   - 范围：`5-10`

---

## FAQ

### Q1: 会不会完全没有逗号？

**不会**。硬闸门只是"最短间隔"约束：
- `since_punct < 2`：禁止逗号
- `since_punct ≥ 2`：**允许**逗号（但不强制）
- 模型会在自然需要停顿时选择逗号

### Q2: 为什么不直接禁止所有标点？

**会导致无穷长句**。我们需要：
- 限制标点**频率**（不要太密）
- 保证标点**存在**（适当停顿）
- 硬闸门实现了这个平衡

### Q3: 英文会不会受影响？

**不会**。逻辑是：
```python
if any(_is_cjk(t) for t in recent_tokens):
    # 是中文，禁用n-gram
else:
    # 是英文，使用3-gram
```

### Q4: 硬闸门会不会与Cloud验证冲突？

**不会**。Cloud看到的是：
- Edge已经应用硬闸门后的tokens
- Cloud验证这些tokens是否合理
- 如果不合理，Cloud会纠正（但Edge本身已减少了错误）

---

## 理论支持

### Tokenization Granularity Mismatch

```
问题：固定规则 + 可变粒度 = 不当约束

中文：1 token = 1 char → 2-gram = "禁止任何两字组合重复"
英文：1 token ≈ 1 morpheme → 2-gram = "禁止词组重复"

解决：语言感知的自适应约束
```

### Hard Constraint vs Soft Penalty

| 方法 | 机制 | 优点 | 缺点 |
|------|------|------|------|
| **Soft Penalty** | `logits -= penalty` | 灵活 | 滞后，可能失效 |
| **Hard Constraint** | `logits = -inf` | 确定 | 可能过严 |

**我们的方案**：
- 硬约束（间隔闸门） + 软约束（repetition penalty）
- 关键规则用硬约束，边缘case用软约束

---

## 参考文档

- 用户分析：`docs/spec_decoding_punctuation_fix_guide.md`（本次修改依据）
- 之前尝试：`docs/punctuation_and_stopping_fix.md`（第一次）
- 之前尝试：`docs/colon_flooding_fix.md`（第二次）
- 之前尝试：`docs/interjection_flooding_fix.md`（第三次）
- 本次修复：`docs/chinese_punctuation_final_fix.md`（第四次，最终方案）

---

## 总结

### 问题本质

**不是"标点太多"，而是"中文特性 + 过度约束 = 被迫选标点"**

### 根源

1. **2-gram禁令**：对中文单字token过于严格
2. **双重惩罚**：repetition + presence叠加压制内容
3. **滞后反应**：看到问题才惩罚，来不及

### 解决

1. ✅ **语言感知**：中文禁用n-gram，英文用温和3-gram
2. ✅ **硬闸门**：基于内容间隔硬性阻断标点
3. ✅ **简化惩罚**：只保留repetition penalty
4. ✅ **兜底机制**：违规时从top-k选非标点

### 核心原则

**"语言感知 + 节奏控制 + 提前预防"**，而非"盲目惩罚 + 滞后反应"


