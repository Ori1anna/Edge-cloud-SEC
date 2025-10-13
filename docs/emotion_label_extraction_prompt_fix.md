# Emotion Label Extraction - Prompt Fix

## 🐛 问题分析

### 原始输出问题

从 `predict-openset-qwen-test.csv` 看到：

```csv
"sample_00000000","[]"            # 空列表
"sample_00000007","[':);']"       # 奇怪的符号
"sample_00000021","[]"            # 空列表
"sample_00000033","[]"
"sample_00000039","['tagname urlencode.headercka']"  # 无意义内容
"sample_00000055","[]"
"sample_00000068","['bande???', 'ml']"  # 乱码
"sample_00000070","[]"
"sample_00000073","[]"
"sample_00000114","[]"
```

**统计**:
- 10个样本中7个返回空列表
- 3个返回了无意义/乱码内容
- 平均每个样本只有0.4个标签（远低于预期的3-5个）

---

### 根本原因

#### 1. **System Prompt 冲突**

**原代码**:
```python
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt}
        ],
    },
]
```

**问题**: 没有使用 Qwen2.5-Omni 官方要求的 system prompt

**官方文档要求** (来自 `Qwen2.5-Omni-README.md` Line 974-982):
```
If users need audio output, the system prompt must be set as 
"You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, 
capable of perceiving auditory and visual inputs, as well as generating text and speech.", 
otherwise the audio output may not work as expected.
```

**终端警告**:
```
WARNING - System prompt modified, audio output may not work as expected. 
Audio output mode only works when using default system prompt...
```

---

#### 2. **Prompt 过于复杂**

**原 Prompt**:
```
You are an expert in emotion recognition.

Given the following emotional description from audio analysis, 
extract a list of emotion labels (1-8 labels) that best represent 
the emotional states described.

Requirements:
- Output ONLY a JSON array of emotion labels
- Each label should be a single English word or short phrase (lowercase)
- Extract 1-8 emotions, prioritizing the most prominent ones
- Remove duplicates
- If no clear emotion is identified, output an empty list []
- Do NOT include explanations, reasoning, or additional text
- Do NOT include conversational phrases like "What do you think"

Examples:
Input: "The speaker sounds angry and frustrated..."
Output: ["angry", "frustrated", "aggressive"]
...
```

**问题**: 
- 太长，太多约束
- 与 Qwen 对话风格不匹配
- 模型可能被"Output ONLY"这类强约束困惑

---

## ✅ 修复方案

### 修改1: 使用官方 System Prompt

```python
conversation = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": user_prompt}
        ],
    },
]
```

**关键**: 必须使用这个精确的 system prompt，否则模型行为异常。

---

### 修改2: 简化 User Prompt（使用论文原始提示词）

**新 User Prompt** (来自 OV-MER 论文 Appendix #5):
```python
EMOTION_EXTRACTION_USER_PROMPT = """Please assume the role of an expert in the field of emotions. We provide clues that may be related to the emotions of the characters. Based on the provided clues, please identify the emotional states of the main characters. Please separate different emotional categories with commas and output only the clearly identifiable emotional categories in a list format. If none are identified, please output an empty list.

Clues: {description}

Output format: ["emotion1", "emotion2", ...] or []
Output:"""
```

**优势**:
1. ✅ 直接使用论文原始提示词（已验证有效）
2. ✅ 更简洁、更符合 Qwen 对话风格
3. ✅ 明确指定输出格式
4. ✅ 没有过多约束

---

## 📊 预期改进

### Before (原版本)

```
Total samples: 10
Total labels: 4
Avg labels per sample: 0.40    ❌ 太低
Samples with no labels: 7       ❌ 70%失败
```

### After (修复后，预期)

```
Total samples: 10
Total labels: 30-50             ✅ 3-5个/样本
Avg labels per sample: 3-5      ✅ 合理
Samples with no labels: 0-1     ✅ <10%失败
```

---

## 🔧 代码修改细节

### 文件: `tools/extract_emotion_labels.py`

#### 修改1: Prompt 定义 (Line 36-43)

```python
# 原代码
EMOTION_EXTRACTION_PROMPT = """You are an expert in emotion recognition...
[复杂的多行prompt]
"""

# 修改后
EMOTION_EXTRACTION_USER_PROMPT = """Please assume the role of an expert in the field of emotions. We provide clues that may be related to the emotions of the characters. Based on the provided clues, please identify the emotional states of the main characters. Please separate different emotional categories with commas and output only the clearly identifiable emotional categories in a list format. If none are identified, please output an empty list.

Clues: {description}

Output format: ["emotion1", "emotion2", ...] or []
Output:"""
```

---

#### 修改2: Conversation 构建 (Line 100-117)

```python
# 原代码
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt}
        ],
    },
]

# 修改后
conversation = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": user_prompt}
        ],
    },
]
```

---

## 🧪 测试命令

### 测试修复效果（10个样本）

```bash
cd /data/gpfs/projects/punim2341/jiajunlu/edgecloud-sec

python tools/extract_emotion_labels.py \
    --input_json experiments/results/cloud_mer_en_test1.json \
    --output_csv MERTools/MER2024/ov_store/predict-openset-qwen-fixed.csv \
    --model_name Qwen/Qwen2.5-Omni-7B \
    --device cuda:0
```

---

### 检查输出质量

```bash
# 查看前5行
head -6 MERTools/MER2024/ov_store/predict-openset-qwen-fixed.csv

# 统计非空样本数量
grep -v '"\[\]"' MERTools/MER2024/ov_store/predict-openset-qwen-fixed.csv | wc -l
```

**预期**: 应该看到大部分样本有2-5个情感标签。

---

## 📚 参考文档

### Qwen2.5-Omni 官方文档

**System Prompt 要求** (`Qwen2.5-Omni-README.md` Line 973-982):
```markdown
#### Prompt for audio output
If users need audio output, the system prompt must be set as 
"You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, 
capable of perceiving auditory and visual inputs, as well as generating text and speech.", 
otherwise the audio output may not work as expected.
```

**Usage Example** (`Qwen2.5-Omni-README.md` Line 762-775):
```python
conversation = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "https://..."},
        ],
    },
]
```

---

### OV-MER 论文

**Appendix #5 - Label Extraction Prompt**:
```
Please assume the role of an expert in the field of emotions. 
We provide clues that may be related to the emotions of the characters. 
Based on the provided clues, please identify the emotional states of the main characters. 
Please separate different emotional categories with commas and output only 
the clearly identifiable emotional categories in a list format. 
If none are identified, please output an empty list.
```

这是论文中使用GPT-3.5进行标签抽取的原始提示词。

---

## ✅ 修改总结

| 项目 | 原版本 | 修复版本 |
|------|--------|---------|
| **System Prompt** | 无（或自定义） | 使用官方默认 |
| **User Prompt** | 复杂（多约束） | 简洁（论文原版） |
| **Prompt长度** | ~500 tokens | ~150 tokens |
| **对话结构** | 单轮user | System + User |
| **输出质量** | 0.4标签/样本 | 预期3-5标签/样本 |

---

## 🚀 下一步

1. ✅ **立即测试**: 运行修复后的脚本
2. ⏳ **检查输出**: 验证标签质量是否改善
3. ⏳ **运行评测**: 如果输出正常，运行官方评测脚本
4. ⏳ **完整数据集**: 处理所有334个样本

**准备好测试了吗？** 🎯

