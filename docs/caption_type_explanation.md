# Caption Type参数说明

## 📋 问题

用户询问：
1. `caption_type`参数有什么用？
2. 模型输入是只有音频，还是音频+字幕？

---

## 🔑 核心答案

### 1. 模型输入：**只有音频**（无字幕）

**证据**（`src/models/cloud_model.py` 第84-98行）：

```python
# Prepare conversation format with audio input
conversation = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen..."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio_waveform},  # ✅ 只有音频
            {"type": "text", "text": prompt}             # ✅ 只有prompt（任务指令）
        ],
    },
]
```

**结论**：
- ✅ 模型输入：**音频 + Prompt（任务指令）**
- ❌ 模型输入：**不包含字幕/文本内容**

**Edge模型也相同**（`src/models/edge_model.py` 第102-116行）：
```python
conversation = [
    ...,
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio_features},  # ✅ 只有音频
            {"type": "text", "text": prompt}             # ✅ 只有prompt
        ],
    },
]
```

---

### 2. `caption_type`参数：**只用于选择reference caption**

**定义**（`experiments/runs/run_cloud_baseline.py` 第47-59行）：

```python
def get_caption_field(sample: Dict[str, Any], caption_type: str, language: str) -> str:
    """Get the appropriate caption field based on type and language"""
    
    if caption_type == "original":
        return sample.get("caption", "")  # 原始标注
    
    elif caption_type == "audio_only":
        if language == "chinese":
            return sample.get("chinese_caption", "")  # 中文audio-only标注
        elif language == "english":
            return sample.get("english_caption", "")  # 英文audio-only标注
    else:
        raise ValueError(f"Unsupported caption type: {caption_type}")
```

**使用场景**（第181行）：

```python
# 从数据集中获取reference caption（用于评估）
reference_caption = get_caption_field(sample, caption_type, language)

# 计算BLEU（对比生成文本和reference caption）
bleu_score = metrics.compute_bleu([reference_caption], generated_text)
```

**结论**：
- ✅ `caption_type`**只影响评估时的reference**（ground truth）
- ❌ `caption_type`**不影响模型输入**（模型看不到字幕）

---

## 📊 完整流程图

### 数据流程

```
Dataset Sample {
    "audio_path": "path/to/audio.wav",
    "caption": "原始标注（可能包含文本内容）",
    "chinese_caption": "中文audio-only标注（纯音频标注）",
    "english_caption": "英文audio-only标注（纯音频标注）"
}

↓

[1] 加载音频
audio_waveform = load_audio(audio_path)  # ✅ 只有音频波形

↓

[2] 准备模型输入
conversation = [
    {role: "user", content: [
        {"type": "audio", "audio": audio_waveform},  # ✅ 音频
        {"type": "text", "text": prompt}             # ✅ 任务指令（如"描述情感"）
    ]}
]
# ❌ 没有字幕/文本内容！

↓

[3] 模型生成
generated_text = model.generate(conversation)
# 模型只"看到"音频和任务指令，没有看到字幕

↓

[4] 评估（选择reference）
if caption_type == "audio_only":
    reference_caption = sample["chinese_caption"]  # 纯音频标注
else:
    reference_caption = sample["caption"]  # 原始标注

BLEU = compare(generated_text, reference_caption)
```

---

## 🔍 Caption Type详解

### Option 1: `caption_type="original"`

**Reference**: 使用 `sample["caption"]`（原始标注）

**特点**：
- 可能包含**文本内容信息**（如"角色说了什么"）
- 可能包含**音频+文本融合**的标注

**示例**（MER2024数据集）：
```json
"caption": "在音频中，角色在一开始时音调上挑，表达出不满与不耐。之后边笑边说，表达出高兴中带着一份调侃。；这句话可能是男性医生对某个问题的回答或者解释。根据音频线索中角色的语调变化..."
```
- ✅ 包含音频线索（"音调上挑"）
- ✅ 包含文本内容（"这句话可能是..."）

---

### Option 2: `caption_type="audio_only"`

**Reference**: 使用 `sample["chinese_caption"]` 或 `sample["english_caption"]`

**特点**：
- **只包含音频信息**（声学、韵律特征）
- **不包含文本内容**（不提及说了什么）

**示例**（MER2024数据集，audio_only标注）：
```json
"chinese_caption": "说话人语速较快，音调起伏较大，音量时强时弱，停顿较短，给人一种激动的感觉。"
```
- ✅ 只描述音频特征（语速、音调、音量）
- ❌ 不包含文本内容（不说"他说了什么"）

---

## 🎯 您的测试配置

### 当前使用的参数

```bash
--caption_type audio_only
--language chinese
```

**效果**：
- **模型输入**: 音频 + Prompt（任务指令）
  - ❌ **不包含字幕**
  - ❌ **不包含文本内容**
- **Reference caption**: `sample["chinese_caption"]`（纯音频标注）

**这是正确的配置！**

---

## 💡 为什么这样设计？

### 场景1: Audio-only任务（您的配置）

**目标**: 测试模型**仅从音频**理解情感的能力

**配置**:
```bash
--caption_type audio_only
```

**效果**:
- 模型只看音频，不看文本
- Reference也是纯音频标注
- **公平对比**：模型生成vs纯音频reference

---

### 场景2: Multimodal任务

**目标**: 测试模型**融合音频+文本**理解情感的能力

**配置**:
```bash
--caption_type original
```

**效果**:
- 模型只看音频（但原始标注可能包含文本信息）
- Reference是音频+文本融合标注
- **不公平**：模型只有音频，但reference有文本信息

**结论**: 对于纯音频任务，应该用 `caption_type="audio_only"`

---

## 📊 对比示例

### Sample 00000000

#### Original Caption
```
"在音频中，角色在一开始时音调上挑，表达出不满与不耐。之后边笑边说，表达出高兴中带着一份调侃。；这句话可能是男性医生对某个问题的回答或者解释。根据音频线索中角色的语调变化，从不满与不耐到高兴中带着一份调侃，我们可以推断这句话可能带有一种幽默或者自嘲的语气。男性医生通过开玩笑和调侃的方式缓解了紧张的话题，表现出一种轻松和愉快的情绪。"
```
- ✅ 包含音频线索
- ✅ **包含文本内容推断**（"这句话可能是..."）
- ✅ 包含角色信息（"男性医生"）

#### Audio-only Caption
```
"说话人语速适中，音调先上扬后下降，音量正常，停顿自然，语气先不耐后调侃，整体情绪从不满转为轻松愉悦。"
```
- ✅ 只有音频特征
- ❌ 不包含文本内容
- ❌ 不推断说了什么

#### 模型生成（只看音频）
```
"说话人的语气听起来有些无奈和困惑，像是在回应别人的问题时发现自己并不清楚答案。他的话语间似乎带着一丝紧张，可能是因为这个问题对他来说确实比较棘手。"
```

### 评估对比

| 配置 | Reference | BLEU | 公平性 |
|------|-----------|------|--------|
| `caption_type="original"` | Original caption（含文本） | 低 | ❌ 不公平 |
| `caption_type="audio_only"` | Audio-only caption（纯音频） | 适中 | ✅ 公平 |

**原因**：
- 模型只能从音频推断情感
- 如果reference包含文本内容（"他说了什么"），模型无法match
- 应该用纯音频标注作为reference

---

## 🎯 总结

### 核心结论

#### Q1: 模型输入是什么？

**答**：**只有音频**（没有字幕）

**证据**：
```python
conversation = [
    {"role": "user", "content": [
        {"type": "audio", "audio": audio_waveform},  # ✅ 音频
        {"type": "text", "text": prompt}             # ✅ Prompt（任务指令）
        # ❌ 没有字幕！
    ]}
]
```

#### Q2: `caption_type`参数的作用？

**答**：**选择评估时的reference caption**

| caption_type | Reference来源 | 特点 |
|--------------|---------------|------|
| `"original"` | `sample["caption"]` | 可能包含文本内容 |
| `"audio_only"` | `sample["chinese_caption"]`<br>`sample["english_caption"]` | 纯音频特征 |

**用途**：
- ✅ 只影响**评估**（选择哪个reference）
- ❌ 不影响**模型输入**（模型看不到）

#### Q3: 您的配置正确吗？

**答**：✅ **完全正确！**

```bash
--caption_type audio_only  # ✅ 用纯音频标注作为reference
--language chinese         # ✅ 中文任务
```

**效果**：
- 模型只看音频生成
- Reference是纯音频标注
- 公平对比

---

## 📝 建议

### 推荐配置

**Audio-only任务**（您的情况）：
```bash
--caption_type audio_only  # ✅ 推荐
```

**Multimodal任务**（如果有字幕作为输入）：
```bash
--caption_type original
# 但需要修改代码，将字幕也传给模型
```

**当前您的配置是最佳实践！** ✅

---

**文档已创建**: `docs/caption_type_explanation.md`

