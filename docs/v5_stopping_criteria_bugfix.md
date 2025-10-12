# V5 Stopping Criteria Bugfix - 字符计数Bug修复

## 问题诊断

### 测试结果分析（newprompt21）

**配置**（正确）：
```python
target_sentences=2      # 要求2句
min_chars=90           # 要求90字符
min_new_tokens=48      # 要求48 tokens
max_new_tokens=128     # 上限128
```

**实际输出**（错误）：

| Sample | Tokens | 实际字符 | 期望字符 | 状态 |
|--------|--------|----------|----------|------|
| 00000000 | 31 | ~50 | 90 | ❌ 提前停止 |
| 00000007 | **124** | ~200 | 90 | ⚠️ 长但无句号 |
| 00000021 | 40 | ~65 | 90 | ❌ 提前停止 |
| 00000033 | 31 | ~50 | 90 | ❌ 提前停止 |
| 其他 | 22-49 | ~35-80 | 90 | ❌ 提前停止 |

**平均**：41.7 tokens，远低于预期的80-120 tokens

---

## 根本原因

### Bug：字符计数包含了Prompt

**错误的代码**（修复前）：
```python
def __call__(self, input_ids, scores, **kwargs):
    self.generated_count += 1
    
    if sentence_count >= n_sentences and tokens >= min_tokens:
        # BUG: Decode ENTIRE input_ids (includes prompt!)
        decoded_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        char_count = len(decoded_text)
        
        if char_count >= self.min_chars:
            return True
```

**实际执行**：
```python
# Sample 00000000
input_ids = [
    prompt_tokens: 352 tokens → "任务：请生成情感说明长句...（约300字）",
    generated_tokens: 31 tokens → "他语速缓慢...（约50字）"
]

# 解码
decoded_text = "任务：请生成...他语速缓慢..."
char_count = len("任务...他语速...") ≈ 350字

# 检查
char_count (350) >= min_chars (90) → True ✅

# 但是：
sentence_count = 1 < n_sentences (2) → False ❌

# 所以不会在第1个句号停止，需要等第2个句号
# 但实际Sample 00000000只有31 tokens就停了！
```

**矛盾！** 让我重新检查...

---

## 深入分析

### 观察：为什么Sample 00000000在31 tokens停止？

让我看Sample 00000000的输出：
```
"他语速缓慢，音调低沉，停顿较多，在讲述自己的经历时，似乎有些无奈和沮丧，可能是因为自己在情感上遇到了挫折。"
```

**只有1个句号！** 

但`n_sentences=2`，理论上不应停止...

### 可能的原因

#### 原因1：Stopping criteria在draft generation中被提前触发

查看代码第1043-1056行：

```python
# In _generate_draft_tokens_incremental
if 'stopping_criteria' in context:
    full_sequence = torch.cat([context['input_ids'][0], torch.tensor(draft_tokens)])
    should_stop = any(criterion(full_sequence, None) for criterion in context['stopping_criteria'])
    if should_stop:
        logger.info("Stopping criteria met at step X, ending draft generation")
        break  # 停止当前draft generation
```

**问题**：
- 每次生成5个draft tokens后检查
- 如果其中包含句号，触发检查
- 如果字符计数bug（包含prompt），可能误判为"已达到min_chars"
- 停止draft generation → 返回空draft_tokens → 主循环break

#### 原因2：`min_new_tokens=48`的检查时机

```python
if sentence_count >= 2 and generated_count >= 48:
    # 检查字符
```

**但我们的配置是`min_new_tokens_sc=48`**，如果第1个句号出现在31 tokens时：
- `generated_count=31 < 48` → 不检查字符
- 继续生成...

**但为什么会停在31？**

让我检查是否有其他min_tokens的检查...

等等！我发现了：`StopWhenNSentences`的`generated_count`是在`__call__`中递增的，**每次调用都+1**！

所以如果第31个token是句号：
- `generated_count=31 < 48` → 不满足
- 不会触发停止检查 ✅

但为什么实际停了？

---

## 真正的问题

让我重新看测试结果...

**关键发现**：
- Sample 00000000: cloud_calls=6, acceptance_rate=1.0
- 只有6次cloud调用，全部接受
- 31 tokens / 6 calls ≈ 5 tokens/call

这说明Edge生成了6个block（每个5 tokens），然后就停止了。

**停止原因**：在第6个block的draft generation中，stopping_criteria被触发！

**但根据`n_sentences=2`的逻辑，不应该在1个句号时停止...**

**等等！** 我需要检查`create_stopping_criteria`中`prompt_type="detailed"`的逻辑！

<function_calls>
<invoke name="read_file">
<parameter name="target_file">/data/gpfs/projects/punim2341/jiajunlu/edgecloud-sec/src/models/stopping_criteria.py

