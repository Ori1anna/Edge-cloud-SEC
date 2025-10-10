# 改进的停止策略：N句到顶停

## 问题分析

### 原始问题
1. **过于严格的"第一句就停"**：遇到第一个`。`就停止，不适合detailed输出
2. **硬截断风险**：依赖`max_new_tokens`可能导致句子被截断
3. **表达不充分**：detailed prompt需要更充分的表达空间

## 新的"N句到顶"停止策略

### 核心特性
- **灵活句子计数**：允许1-2句完整表达，而不是强制单句
- **最小长度保护**：`min_new_tokens=32`，避免刚起句就停止
- **Token-level精确控制**：基于token ID而非字符串匹配，避免BPE问题
- **Prompt类型适配**：根据prompt类型自动调整参数

### 实现细节

#### 1. StopWhenNSentences类
```python
class StopWhenNSentences(StoppingCriteria):
    def __init__(self, tokenizer, n_sentences=2, sentence_end_chars=("。", "."), min_new_tokens=32):
        # 支持N句停止，默认2句
        # 最小32个token保护
        # 基于token ID的精确匹配
```

#### 2. Prompt类型适配
- **detailed**: `n_sentences=2`, `min_new_tokens=32` (允许更充分表达)
- **concise**: `n_sentences=1`, `min_new_tokens=24` (保持简洁)
- **default**: 使用传入参数

#### 3. 配置参数
```python
# 详细但不强卡一句的推荐配置
stopping_criteria = create_stopping_criteria(
    tokenizer=tokenizer,
    n_sentences=2,                    # 允许2句
    sentence_end_chars=("。", "."),   # 只计数句末标点
    min_new_tokens=32,               # 最小32个token
    prompt_type="detailed"           # 根据prompt类型调整
)

# 生成参数
generation_kwargs = {
    'max_new_tokens': 120,           # 充足的长度空间
    'do_sample': False,              # 确定性解码
    'no_repeat_ngram_size': 2,       # 防止重复
    'repetition_penalty': 1.05,      # 轻微重复惩罚
    'stopping_criteria': stopping_criteria  # 使用N句停止策略
}
```

## 优势对比

### 原始策略 vs 新策略

| 特性 | 原始策略 | 新策略 |
|------|----------|--------|
| 停止条件 | 第一个`。`就停 | N句完整后停止 |
| 最小保护 | 24 tokens | 32 tokens |
| 表达空间 | 受限单句 | 允许1-2句充分表达 |
| 截断风险 | 依赖max_tokens硬截断 | 自然句末停止 |
| 适配性 | 固定策略 | 根据prompt类型调整 |

## 预期效果

### Detailed Prompt输出
- **更充分的表达**：允许2句完整描述，包含"主要情绪 + 声学线索"
- **自然停止**：在第二个`。`后自然停止，不会硬截断
- **长度适中**：通常30-70个汉字，符合detailed要求

### 质量提升
- **消除截断**：避免"停顿顿停了"这种不完整句子
- **减少语无伦次**：完整句子结构，逻辑连贯
- **保持专业性**：符合情感分析任务的输出格式

## 技术实现要点

### 1. Token ID精确匹配
```python
# 获取句末标点的token ID
sentence_end_ids = []
for char in sentence_end_chars:
    tokens = tokenizer.encode(char, add_special_tokens=False)
    if tokens:
        sentence_end_ids.append(tokens[-1])
```

### 2. 句子计数逻辑
```python
# 遇到句末标点时计数
if last_token_id in self.sentence_end_ids:
    self.sentence_count += 1
    
    # 达到目标句数且满足最小token要求时停止
    if (self.sentence_count >= self.n_sentences and 
        self.generated_count >= self.min_new_tokens):
        return True
```

### 3. EOS处理
```python
# EOS token始终停止（模型级停止）
if last_token_id in self.eos_ids:
    return True
```

## 使用建议

### 测试命令
```bash
python experiments/runs/run_speculative_decoding_cpu_limited.py \
  --dataset_path data/processed/mer2024/manifest_audio_only_final.json \
  --output_name speculative_decoding_mer_chinese_improved \
  --max_samples 100 \
  --verbose \
  --caption_type audio_only \
  --language chinese \
  --prompt_type detailed \
  --entropy_threshold 1.5 \
  --k 5 \
  --max_memory_gb 16.0
```

### 预期输出特征
- **完整句子**：以`。`结尾的完整中文句子
- **适当长度**：30-70个汉字，符合detailed要求
- **逻辑连贯**：包含情感和声学线索的连贯描述
- **自然停止**：在第二句结束后自然停止

这个策略完全符合您提出的"详细但不强卡一句"的要求，既保证了充分表达，又避免了无限跑题的问题。
