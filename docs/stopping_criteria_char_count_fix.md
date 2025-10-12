# Stopping Criteria Character Count Fix - 修复字符计数Bug

## 问题诊断

### 测试结果：speculative_decoding_mer_chinese_newprompt21.json

**配置**：
```python
target_sentences=2
min_chars=90
min_new_tokens=48
max_new_tokens=128
```

**实际输出**：

| Sample | Tokens | 句号数 | 字符数（估算） | 状态 |
|--------|--------|--------|----------------|------|
| 00000000 | 31 | 1 | ~50 | ❌ 提前停止 |
| 00000007 | **124** | 0 | ~200 | ✅ 长但无句号 |
| 00000021 | 40 | 1 | ~65 | ❌ 提前停止 |
| 00000033 | 31 | 2 | ~50 | ❌ 提前停止 |
| 00000039 | 22 | 1 | ~35 | ❌ 提前停止 |
| 其他 | 29-49 | 1-2 | ~45-80 | ❌ 提前停止 |

**矛盾现象**：
- ✅ 配置要求：`min_chars=90`
- ❌ 实际输出：大部分只有30-50字符
- ❓ 为什么会在远未达到90字符时就停止？

---

## 根本原因

### Bug：`min_chars`计算包含了prompt内容

**当前代码**（`src/models/stopping_criteria.py`，第89-100行）：

```python
def __call__(self, input_ids, scores, **kwargs):
    if sentence_count >= n_sentences and tokens >= min_tokens:
        # Bug: Decode the ENTIRE input_ids (includes prompt!)
        decoded_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        char_count = len(decoded_text)
        
        if char_count >= self.min_chars:  # 检查总字符数
            return True
```

**实际执行**：

```python
# Sample 00000000 生成31 tokens后遇到第1个句号
input_ids = [prompt_tokens (300+), generated_tokens (31)]

# 解码整个序列
decoded_text = "任务：请生成情感说明...（300+字） 他语速缓慢...（50字）"
char_count = 350+  # 包含了prompt！

# 检查
char_count (350) >= min_chars (90) → True ✅
sentence_count (1) >= n_sentences (2) → False ❌

# 结果：虽然char_count满足，但sentence_count不满足，继续生成
```

**等等，这不对！** 让我重新看逻辑...

### 实际问题：`n_sentences=2`的逻辑

```python
if sentence_count >= 2 and tokens >= 48:
    # 检查字符数
    if new_chars >= 90:
        return True
```

**现在的行为**：
- 生成第1个句号：`sentence_count=1 < 2` → 不停止 ✅
- 生成第2个句号：`sentence_count=2 ≥ 2` → 检查其他条件
- 如果`tokens≥48`且`chars≥90` → 停止

**问题**：大部分样本在30-40 tokens时就遇到第1个句号，然后**立即停止**了！

这说明：**停止不是由stopping_criteria触发的**，而是由其他逻辑！

让我再检查`_is_eos_token`中是否有句号：

啊！我之前看到`_is_eos_token`中**移除了句号**（第126行注释），但让我检查是否真的完全移除了：

<function_calls>
<invoke name="grep">
<parameter name="pattern">1773

