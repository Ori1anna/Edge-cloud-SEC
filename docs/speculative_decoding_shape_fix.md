# Speculative Decoding Shape Error 修复

## 错误分析

### 错误信息
```
IndexError: tuple index out of range
File ".../modeling_qwen2_5_omni.py", line 2364
if input_ids is not None and input_ids.shape[1] != 1:  # Prefill stage
```

### 根本原因
`current_input_ids`的形状不正确，导致Qwen2.5-Omni模型的thinker方法无法正确处理输入。

### 问题定位
1. **last_token_id形状问题**：
   ```python
   # 问题代码
   last_token_id = context['input_ids'][0, -1:].clone()  # Shape: [1]
   current_input_ids = context['last_token_id']  # 形状不匹配！
   ```

2. **期望的形状**：
   - Qwen2.5-Omni thinker期望`input_ids`形状为`[batch_size, seq_len]`
   - 对于增量生成，应该是`[1, 1]`

## 修复方案

### 修复1：正确的last_token_id形状
```python
# 修复前
last_token_id = context['input_ids'][0, -1:].clone()  # Shape: [1]

# 修复后
last_token_id = context['input_ids'][0, -1:].clone().unsqueeze(0)  # Shape: [1, 1]
```

### 修复2：current_input_ids形状检查
```python
# 修复前
current_input_ids = context['last_token_id']  # 可能形状错误

# 修复后
if 'last_token_id' in context and context['last_token_id'] is not None:
    last_token = context['last_token_id']
    if last_token.dim() == 1:
        current_input_ids = last_token.unsqueeze(0)  # Shape: [1, 1]
    else:
        current_input_ids = last_token
```

### 修复3：增强调试信息
```python
# 添加详细的形状调试信息
logger.debug(f"Step {step+1}: input_ids shape={current_input_ids.shape}, value={current_input_ids.item()}")
logger.debug(f"Step {step+1}: past_key_values present: {current_past_key_values is not None}")
logger.debug(f"Step {step+1}: step_inputs keys: {list(step_inputs.keys())}")
```

## 修复效果

### 修复前
```
IndexError: tuple index out of range
→ 程序崩溃，无法继续执行
```

### 修复后
```
正确形状的input_ids → Qwen2.5-Omni thinker正常处理 → 继续增量生成
```

## 技术细节

### 形状要求
- **Prefill阶段**：`input_ids.shape[1] > 1`
- **增量生成**：`input_ids.shape[1] == 1`
- **Batch维度**：始终为`[batch_size, seq_len]`

### 关键修复点
1. **last_token_id初始化**：确保形状为`[1, 1]`
2. **current_input_ids设置**：添加形状检查和修正
3. **调试信息**：帮助诊断形状问题

这个修复解决了speculative decoding中的tensor形状不一致问题，确保Qwen2.5-Omni模型能够正确处理增量生成的输入。
