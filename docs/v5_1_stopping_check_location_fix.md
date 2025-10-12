# V5.1 Stopping Check Location Fix - 修复停止检查位置Bug

## 问题诊断

### 测试结果（newprompt22）

**配置（正确）**：
```python
target_sentences=2, min_chars=90, min_new_tokens=48, max_new_tokens=128
```

**实际输出（仍然错误）**：
- 9/10样本只有1句话（19-49 tokens）
- 平均：41.7 tokens
- 只有1个样本（00000007）生成了124 tokens（但无句号）

**Bug已修复但问题仍在**：
- ✅ `generated_count`修复（基于序列长度）
- ✅ `char_count`修复（只计算新生成部分）
- ❌ **仍然只有1句话**

---

## 根本原因

### 新发现的Bug：Stopping criteria在错误的位置检查

**问题代码**（`src/speculative_decoding.py`，第1043-1056行）：

```python
# In _generate_draft_tokens_incremental() - WRONG LOCATION!
for step in range(k):  # 生成k=5个draft tokens
    ...
    draft_tokens.append(next_token)
    
    # Check stopping criteria HERE - 在draft generation中检查
    if 'stopping_criteria' in context:
        full_sequence = cat([context['input_ids'], draft_tokens])
        should_stop = any(criterion(full_sequence, None))
        if should_stop:
            break  # 跳出draft generation循环
            
    # 继续下一个token...

# 返回draft_tokens（可能只有1-2个token，因为提前break了）
return draft_tokens
```

**问题流程**：

```python
# 主循环调用draft generation
draft_tokens = _generate_draft_tokens_incremental(context, k=5)

# Draft generation内部：
Step 1: 生成token A
Step 2: 生成token B  
Step 3: 生成token C
Step 4: 生成"。" (句号)
  → stopping_criteria检查
  → sentence_count=1 < 2，继续？
  → 但由于某种原因break了！
  → 返回draft_tokens=[A, B, C, 。] (只有4个，不是5个)

# 回到主循环
if not draft_tokens or len(draft_tokens) < 期望值:
    break  # 主循环也停止

# 结果：只生成了30-40 tokens就停止
```

---

## 为什么会提前break？

### 可能原因1：Draft generation中的break逻辑

查看代码第1055行：
```python
if should_stop:
    logger.info(f"Stopping criteria met at step {step+1}, ending draft generation")
    break  # 直接跳出draft循环
```

**即使stopping criteria返回False（因为sentence_count=1<2）**，但如果：
1. Draft generation返回了包含句号的tokens
2. 主循环检查EOS（第607-611行）
3. 发现句号...

**等等！** `_is_eos_token(1773)`应该返回False（句号不是EOS）

让我重新思考...

### 真正的原因：Draft generation被提前break的其他情况

可能是draft generation中的其他检查导致break？让我检查draft generation中所有的break：

<function_calls>
<invoke name="grep">
<parameter name="pattern">break

