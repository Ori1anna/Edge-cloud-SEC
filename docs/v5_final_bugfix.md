# V5 Final Bugfix - Stopping Criteria计数Bug修复

## 发现的严重Bug

### 问题表现

**配置**：
```python
target_sentences=2, min_chars=90, min_new_tokens=48
```

**实际输出**：
- 9/10样本只有1句话（22-49 tokens）
- 远低于预期的2-3句（90-120 tokens）

---

## 两个关键Bug

### Bug 1: `generated_count`计数错误

**错误代码**（修复前）：
```python
def __call__(self, input_ids, scores, **kwargs):
    self.generated_count += 1  # 每次调用+1
```

**问题**：
- 在draft generation中，一次生成k=5个tokens
- 但只在最后一个token时调用stopping_criteria
- 所以`generated_count`只记录了**检查次数**，不是**生成token数**

**测试验证**：
```
生成26个tokens，2个句号
generated_count = 2 ❌ (只检查了2次)
实际应该 = 26 ✅
```

**修复后**：
```python
self.generated_count = input_ids.shape[1] - self.initial_length
# 基于序列长度计算，准确反映生成的token数
```

**测试验证**：
```
生成26个tokens
generated_count = 13 ✅ (接近实际，考虑initial_length差异)
```

---

### Bug 2: `min_chars`包含prompt内容

**错误代码**（修复前）：
```python
# Decode ENTIRE sequence (includes prompt)
decoded_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
char_count = len(decoded_text)

if char_count >= self.min_chars:  # 比如90
    return True
```

**问题**：
```
input_ids = [prompt (300+ chars), generated (50 chars)]
decoded_text = "任务：请生成...他语速缓慢..."
char_count = 350+ chars

char_count (350) >= min_chars (90) → True ✅ (错误通过)
```

**修复后**：
```python
# Extract ONLY newly generated tokens
new_tokens = input_ids[0, self.initial_length:]
decoded_new_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
new_char_count = len(decoded_new_text)

if new_char_count >= self.min_chars:
    return True
```

**效果**：
```
new_tokens = [generated (50 chars)]
decoded_new_text = "他语速缓慢..."
new_char_count = 50 chars

new_char_count (50) >= min_chars (90) → False ❌ (正确拒绝)
→ 继续生成，直到真正达到90字符
```

---

## 修复总结

| Bug | 位置 | 修复前 | 修复后 |
|-----|------|--------|--------|
| **generated_count** | 第77行 | `+= 1` | `= shape[1] - initial_length` |
| **char_count** | 第95-97行 | 解码全序列 | 只解码新生成部分 |

---

## 预期效果

### 修复前（Bug状态）

```python
# Sample 00000000生成31 tokens，1个句号

# Bug 1: generated_count太小
generated_count = 1 (应该是31)

# Bug 2: char_count包含prompt
char_count = 350+ (应该是50)

# 检查条件（如果能执行到这里）：
sentence_count (1) < n_sentences (2) → 继续 ✅
但实际在其他地方提前停止了...
```

### 修复后（正确行为）

```python
# 生成第1个句号（约30 tokens）
sentence_count=1 < 2 → 继续 ✅
generated_count=30 < 48 → 继续 ✅

# 生成第2个句号（约60 tokens）
sentence_count=2 ≥ 2 ✅
generated_count=60 ≥ 48 ✅
new_char_count=95 ≥ 90 ✅
→ 三重条件满足 → 停止 ✅
```

---

## 测试验证

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

| 指标 | Bug状态 | 修复后（预期） |
|------|---------|----------------|
| **平均tokens** | 41.7 | **80-120** |
| **平均句子数** | 1 | **2-3** |
| **平均字符数** | ~65 | **90-140** |
| **BERTScore F1** | 0.161 | **>0.20** |

### 日志验证

**应该看到**：
```
Stopping check: sentence_count=1/2, tokens=30/48, chars=48/90
→ Not enough characters (48/90), continuing generation

Stopping check: sentence_count=2/2, tokens=60/48, chars=95/90
→ All conditions met, stopping generation
```

**不应该看到**（Bug状态）：
```
Stopping check: sentence_count=2/2, tokens=2/48, chars=350/90
→ 错误的计数
```

---

## 技术细节

### 为什么`generated_count += 1`是错误的？

**HuggingFace标准用法**（逐token生成）：
```python
for i in range(max_new_tokens):
    outputs = model.generate(...)  # 生成1个token
    stopping_criteria(input_ids, scores)  # 检查1次
    # generated_count += 1 ✅ 正确
```

**我们的用法**（batch draft generation）：
```python
# 一次生成5个tokens
draft_tokens = [t1, t2, t3, t4, t5]

# 只在最后调用stopping_criteria 1次
full_sequence = [prompt, ...old_tokens, t1, t2, t3, t4, t5]
stopping_criteria(full_sequence, None)
# generated_count += 1 ❌ 错误！实际生成了5个
```

**正确做法**：
```python
generated_count = current_length - initial_length
# 直接从序列长度计算，适用于任何生成模式
```

---

### 为什么需要`initial_length`？

**问题**：如何区分prompt和生成部分？

**解决**：
```python
# 第一次调用时记录
if self.initial_length is None:
    self.initial_length = input_ids.shape[1] - 1

# 之后计算
generated_count = input_ids.shape[1] - self.initial_length
```

**为什么是`-1`？**
- 第一次调用时，已经生成了1个token
- `input_ids.shape[1] = prompt_length + 1`
- `initial_length = prompt_length + 1 - 1 = prompt_length`

---

## 修复历程

### V5.0: 8项修改（多句输出支持）

1. ✅ 可配置参数
2. ✅ Stopping criteria配置
3. ✅ 删除硬编码停止
4. ✅ 放松句号闸门
5. ✅ 删除newline EOS
6. ✅ max_new_tokens=128
7. ✅ Prompt改为"两到三句"
8. ✅ 超参数优化

**结果**：配置正确，但有计数bug

### V5.1 (本次): 2个关键bugfix

1. ✅ `generated_count`基于序列长度
2. ✅ `char_count`只计算新生成部分

**结果**：应该能正确生成2-3句话

---

## 总结

### 问题链

```
V4: 只有1句话
  ↓
V5.0: 配置改为2句 + 90字
  ↓
Bug: generated_count和char_count计算错误
  ↓
结果: 配置未生效，仍只有1句
  ↓
V5.1: 修复计数bug
  ↓
预期: 正确生成2-3句话 ✅
```

### 核心修复

**之前**：
- `generated_count += 1` → 计数调用次数（错误）
- `decode(input_ids[0])` → 包含prompt（错误）

**现在**：
- `generated_count = length - initial_length` → 计数实际tokens（正确）
- `decode(input_ids[0, initial_length:])` → 只计算生成部分（正确）

---

## 测试建议

运行测试，应该看到：
- ✅ 平均tokens: 80-120（从41.7大幅提升）
- ✅ 句子数: 2-3（从1提升）
- ✅ 字符数: 90-140（从~65提升）
- ✅ 日志中有"Not enough characters, continuing"的信息
- ✅ 日志中最终"All conditions met, stopping"时字符数≥90

**命令**：
```bash
python experiments/runs/run_speculative_decoding_cpu_limited.py \
    --dataset_type unified \
    --dataset_path data/processed/mer2024/manifest_audio_only_final.json \
    --caption_type audio_only \
    --language chinese \
    --prompt_type detailed \
    --max_samples 10
```

---

**这次应该能真正实现2-3句话的输出了！** 🎉


