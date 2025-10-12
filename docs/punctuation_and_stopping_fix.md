# Punctuation Flooding & Premature Truncation Fix

## 问题诊断

根据测试结果 `speculative_decoding_mer_chinese_newprompt11.json` 的分析，发现两个主要问题：

### 问题1：标点泛滥（Token ID 3837 = 逗号）

**表现**：
- 生成文本中逗号密集出现
- 例如："你，明知道，龙，舟，队，的，队员..."
- 日志显示token 3837（逗号）反复被低熵策略接受

**根本原因**：
1. **逗号高置信度**：标点符号在语言模型中通常具有极高置信度（低熵）
2. **低熵策略漏洞**：低于阈值的block被全量接受，标点被当作"安全token"
3. **缺乏标点频控**：没有针对标点的抑制/频率控制机制

### 问题2：过早截断（Token ID 1773 = 句号）

**表现**：
- 输出被过早截断，明显未完成
- 例如："...可能是在担心" （句子明显未说完）
- 日志显示："Stop condition met in accepted token 1773"

**根本原因**：
1. **`_is_eos_token`错误**：将句号`。`作为立即停止条件
2. **停止策略不完善**：只检查句号出现，未同时验证长度下限
3. **缺少字符数约束**：`min_new_tokens`只控制token数，未考虑实际字符数

---

## 解决方案

### 修改1：标点频控机制

**文件**：`src/speculative_decoding.py`
**位置**：第852-871行（在repetition penalty之后）

**实现**：
```python
# Punctuation frequency control (anti-punctuation flooding)
punctuation_tokens = {3837: '，', 1773: '。', 30544: '、'}
recent_n = 5  # Check last 5 tokens
punct_threshold = 2  # Max 2 punctuations in last 5 tokens

# Count punctuation in recent draft tokens
recent_tokens_check = draft_tokens[-recent_n:] if len(draft_tokens) >= recent_n else draft_tokens
punct_count = sum(1 for t in recent_tokens_check if t in punctuation_tokens)

# If too many punctuations, strongly suppress them
if punct_count >= punct_threshold:
    for punct_id in punctuation_tokens.keys():
        logits_temp[punct_id] -= 10.0  # Strong suppression
    logger.debug(f"Punctuation suppression: {punct_count}/{recent_n} punctuations, applying -10.0 penalty")

# Always apply moderate punctuation penalty to prevent over-use
punct_base_penalty = 2.0
for punct_id in punctuation_tokens.keys():
    logits_temp[punct_id] -= punct_base_penalty
```

**机制**：
- **滑动窗口检测**：检查最近5个token中的标点数量
- **阈值触发**：如果标点数≥2，施加强惩罚（-10.0）
- **基础抑制**：始终对所有标点施加中度惩罚（-2.0）

---

### 修改2：移除句号立即停止

**文件**：`src/speculative_decoding.py`
**位置**：第111-129行（`_is_eos_token`方法）

**修改前**：
```python
if token_text in ['。', '\n', 'Human', ...]:
    return True
```

**修改后**：
```python
# NOTE: We DO NOT stop on '。' here - sentence ending is handled by stopping_criteria
# This prevents premature truncation before min_chars/min_tokens are reached
if token_text in ['\n', 'Human', 'Human:', ...]:  # 移除'。'
    return True
```

**原因**：
- 将句号检测交给`StopWhenNSentences`统一处理
- 避免绕过长度下限的提前终止

---

### 修改3：双条件停机策略

**文件**：`src/models/stopping_criteria.py`
**位置**：第17-31行（`__init__`）、第80-103行（`__call__`）

**增强参数**：
```python
def __init__(self, tokenizer, n_sentences=2, sentence_end_chars=("。", "."), 
             min_new_tokens=32, min_chars=70):
    self.tokenizer = tokenizer
    self.min_chars = min_chars  # 新增：最小字符数要求
    ...
```

**三重条件停机**：
```python
if self.sentence_count >= self.n_sentences and self.generated_count >= self.min_new_tokens:
    # Decode current sequence to check character count
    decoded_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
    char_count = len(decoded_text)
    
    # Only stop if we've also generated enough characters
    if char_count >= self.min_chars:
        return True
    else:
        # Not enough characters yet, continue generation
        pass
```

**逻辑**：
1. ✅ 句子数 ≥ `n_sentences`（例如2句）
2. ✅ Token数 ≥ `min_new_tokens`（例如32个token）
3. ✅ 字符数 ≥ `min_chars`（例如70个字符）

**只有三个条件全部满足才停止！**

---

### 修改4：低熵标点滥用检测

**文件**：`src/speculative_decoding.py`
**位置**：第424-435行（uncertainty检查后）

**实现**：
```python
# Punctuation flooding detection
punctuation_token_ids = {3837, 1773, 30544}  # 逗号、句号、顿号
punct_count = sum(1 for t in draft_tokens if t in punctuation_token_ids)
punct_ratio = punct_count / len(draft_tokens) if draft_tokens else 0

# If >40% of tokens are punctuation, force cloud verification regardless of entropy
if punct_ratio > 0.4 and not needs_cloud_verification:
    logger.warning(f"Punctuation flooding detected: {punct_count}/{len(draft_tokens)} tokens ({punct_ratio:.1%}) are punctuation")
    logger.warning("Forcing cloud verification to prevent punctuation abuse")
    needs_cloud_verification = True
    max_uncertainty = self.entropy_threshold + 0.1  # Force above threshold
```

**机制**：
- **标点比例检测**：计算draft block中标点token的占比
- **强制升云**：如果标点>40%，即使熵值低也强制Cloud验证
- **双重保险**：与draft阶段的频控配合，形成两层防护

---

## 修改总结

| 修改项 | 文件 | 行号 | 目的 |
|--------|------|------|------|
| 标点频控 | `src/speculative_decoding.py` | 852-871 | 防止draft阶段标点泛滥 |
| 移除句号停止 | `src/speculative_decoding.py` | 117-119 | 交给停止策略统一处理 |
| 双条件停机 | `src/models/stopping_criteria.py` | 17-31, 80-103 | 增加字符数约束 |
| 标点滥用检测 | `src/speculative_decoding.py` | 424-435 | 强制标点过多block升云 |

---

## 预期效果

### 问题1（标点泛滥）解决：
- ✅ **Draft阶段频控**：最近5个token内标点≤2
- ✅ **基础惩罚**：所有标点logits -2.0
- ✅ **强制升云**：标点>40%的block必须Cloud验证

**预期输出**：
```
修改前："你，明知道，龙，舟，队，的，队员，现在，都..."
修改后："你明知道龙舟队的队员现在都还是群菜鸟，..."
```

### 问题2（过早截断）解决：
- ✅ **三重条件**：句子数 + token数 + 字符数
- ✅ **最小长度**：至少70个字符（detailed prompt）
- ✅ **完整句子**：确保句子说完整

**预期输出**：
```
修改前："...可能是在担心"（截断）
修改后："...可能是在担心自己无法胜任这个任务，所以显得有些紧张。"（完整）
```

---

## 测试建议

1. **运行测试**：
```bash
python experiments/runs/run_speculative_decoding_cpu_limited.py \
    --dataset_type unified \
    --dataset_path data/processed/mer2024/manifest_audio_only_final.json \
    --caption_type audio_only \
    --language chinese \
    --prompt_type detailed \
    --max_samples 10
```

2. **观察指标**：
   - 逗号密度：`generated_text`中逗号出现频率
   - 输出长度：`output_tokens`平均值
   - 字符数：`len(generated_text)`平均值
   - 完整度：是否有明显截断

3. **日志关键字**：
   - `"Punctuation suppression"` - 触发标点抑制
   - `"Punctuation flooding detected"` - 触发强制升云
   - `"Stopping criteria met"` - 正常停止
   - 查看最终字符数是否≥70

---

## Token ID 参考

| 标点 | Token ID | 说明 |
|------|----------|------|
| ，   | 3837     | 中文逗号（主要泛滥对象）|
| 。   | 1773     | 中文句号（过早停止触发器）|
| 、   | 30544    | 中文顿号 |

---

## 理论依据

### 为什么标点会高置信度？
1. **语法规则**：标点位置相对固定（句末、短语间）
2. **训练数据**：标点出现频率极高，模型"记住"了模式
3. **低信息量**：标点不承载语义，熵值天然较低

### 为什么需要字符数约束？
1. **Token粒度不一致**：中文1个字可能=1 token，英文1个词可能=多个token
2. **标点占位**：标点也算token，但不增加实际内容
3. **任务需求**：音频描述任务需要"充分描述"，字符数更能反映描述详细度

### 双层防护策略：
- **Layer 1（Draft阶段）**：频控 + 惩罚 → 降低标点生成概率
- **Layer 2（Verification阶段）**：滥用检测 → 强制Cloud纠正

---

## 与之前修改的关系

本次修改**不冲突**于之前的优化：
- ✅ 保留贪心解码（与baseline一致）
- ✅ 保留repetition penalty（1.1）
- ✅ 保留no-repeat 2-gram约束
- ✅ 保留KV cache管理逻辑

**增量改进**：
- 在现有penalties基础上，专门针对**标点**增加约束
- 在现有停止策略基础上，增加**字符数**维度

---

## 潜在风险与缓解

### 风险1：标点过度抑制 → 缺少必要标点
**缓解**：
- 只在"最近5个token内≥2个标点"时触发强抑制
- 正常情况下只有-2.0的基础惩罚（适度）

### 风险2：字符数约束过严 → 生成过长
**缓解**：
- `min_chars=70`是下限，不是目标值
- 仍有`max_new_tokens=120`作为上限
- 停止策略仍优先响应句号（只要满足三重条件）

### 风险3：强制升云增加延迟
**缓解**：
- 只有标点>40%的极端case才触发
- 正常block（标点<40%）不受影响
- 是质量保障，不是常规路径


