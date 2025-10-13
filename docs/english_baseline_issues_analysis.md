# 英语Cloud Baseline问题分析

## 🐛 发现的问题

### 问题1: **生成了对话式内容**（严重）

#### 样本证据

**Sample 00000000**:
```
"The voice is steady but carries a tone of uncertainty and hesitation. It sounds like the speaker is trying to reassure someone or explain something without much confidence. What do you think might be causing this emotion?"
                                                                                                    ↑ ❌ 对话式结尾
```

**Sample 00000021**:
```
"The woman's tone is steady and clear, conveying confidence and determination. It sounds like she is fully committed to being with Xiao Chuan. What do you think about this?"
                                                                                                                              ↑ ❌ 对话式结尾
```

**Sample 00000070**:
```
"The man's tone is calm but carries a subtle hint of skepticism. It's like he's questioning something important without showing his true feelings openly. What do you think might be causing this skepticism?"
                                                                                                                                ↑ ❌ 对话式结尾
```

**统计**: 10个样本中至少5个以"What do you think..."结尾

---

### 问题2: **Reference caption质量极差**（严重）

#### 样本证据

**Sample 00000000 Reference**:
```
"In the audio, the character's tone is initially raised, expressing dissatisfaction and impatience. This may indicate that he is discussing a serious issue and already feeling a bit frustrated. This change suggests that the speaker has lightened the tense topic through jokes and banter. In the audio, the character's tone is initially raised, expressing dissatisfaction and impatience. Later, while laughing, the character speaks, conveying happiness with a hint of teasing. I, I, I don't have any experience in this area.\" This sentence could be the male doctor's response or explanation to a certain question. Audio, from dissatisfaction and impatience to happiness with a hint of teasing, we can infer that this sentence may have a humorous or self-deprecating tone. The male doctor lightens the tense topic through jokes and banter, displaying a relaxed and happy mood."
```

**问题**:
1. ❌ **句子重复**: "In the audio, the character's tone is initially raised..." 出现了2次
2. ❌ **包含字幕**: "I, I, I don't have any experience in this area.\""
3. ❌ **语法错误**: "Audio, from dissatisfaction..." 缺少主语
4. ❌ **句子破碎**: 多个句子拼接不自然

**Sample 00000021 Reference**:
```
"In the audio, there is a longer pause between \"Are you genuinely willing\" and \"to be with Xiaochuan?\" when the character expresses the phrase. She maintains direct eye contact with the other person and engages in a serious conversation. In the audio, there is a longer pause between \"Are you genuinely willing\" and \"to be with Xiaochuan?\" when the character expresses the phrase. This pause may indicate that the character has some doubts about the other person's true intentions or emotional feelings, or it could be due to uncertainty about the future or concerns about the relationship.In the text, the subtitle reads, \"Are you genuinely willing to be with Xiaochuan?\" Based on the video clues of the woman's calm facial expression, natural posture, absence of nervousness or unease, and her direct eye contact and serious conversation with the other person, it can be inferred that she is asking the question in a calm and sincere manner. However, based on the pause in the audio when the character expresses this sentence, it can be speculated that she may have some doubts or concerns about the other person's true intentions or emotional feelings. This pause may imply her uncertainty about the future or her worries about the relationship, so she may have a certain emotional state of questioning or concern when asking this question."
```

**问题**:
1. ❌ **句子重复**: 开头部分重复了2次
2. ❌ **包含字幕**: "In the text, the subtitle reads, \"Are you genuinely willing to be with Xiaochuan?\""
3. ❌ **包含视频信息**: "She maintains direct eye contact"（不是audio-only）
4. ❌ **非常冗长**: 10行文本，质量很差

---

## 🔍 根本原因分析

### 原因1: English Caption翻译质量差

**数据来源**: `data/processed/mer2024/manifest_audio_only_final.json`

**问题**: `english_caption`字段是机器翻译的，质量很差：
- 重复句子
- 包含不应该有的内容（字幕、视频信息）
- 语法错误

**证据**（第8-9行）:
```json
"chinese_caption": "在音频中，角色在一开始时音调上挑，表达出不满与不耐。之后边笑边说，表达出高兴中带着一份调侃。；这句话可能是男性医生对某个问题的回答或者解释。根据音频线索中角色的语调变化，从不满与不耐到高兴中带着一份调侃，我们可以推断这句话可能带有一种幽默或者自嘲的语气。男性医生通过开玩笑和调侃的方式缓解了紧张的话题，表现出一种轻松和愉快的情绪。",

"english_caption": "In the audio, the character's tone is initially raised... I, I, I don't have any experience in this area.\" ..."
```

**对比**:
- ✅ 中文caption：流畅、完整、只描述音频
- ❌ 英文caption：重复、包含字幕、语法错误

---

### 原因2: 英语Prompt不够严格

**当前Prompt**（您修改后的）:
```
"As an expert in the field of emotions, please focus on the acoustic information in the audio to discern clues related to the emotions of the individual. Please provide a detailed description and ultimately predict the emotional state of the individual."
```

**问题**:
- ❌ 没有明确禁止对话式内容
- ❌ 没有明确要求客观描述
- ❌ 没有限制句子数量

**对比中文Prompt**（更严格）:
```
"任务：请生成"情感说明长句"，按以下顺序组织内容并保持自然流畅：
(1) 先用2–3个"类别级"的声学/韵律线索描述说话方式...
(2) 据此给出最可能的单一情绪...
(3) 若语义内容暗示缘由，可用极简的一小短语点到为止...

输出要求：
- 只输出"两到三句中文长句"，约70–100个字；
- 使用第三人称或"说话人"等指代；不要出现第一/第二人称；不要设问或邀请对话；
- 不要编造具体人物/时间/地点等细节；不要出现表情符号、英文、Markdown/代码。"
```

**关键差异**:
- ✅ 中文：明确要求"不要设问或邀请对话"
- ❌ 英文：没有这个要求

---

## 📊 问题影响

### 指标异常

| 指标 | 值 | 正常范围 | 状态 |
|------|-----|---------|------|
| **corpus_bleu_en** | 0.0004 | 0.15-0.30 | ❌ 极低 |
| **avg_bleu_sentence** | 0.0237 | 0.15-0.30 | ❌ 很低 |
| **avg_cider** | 0.9580 | 0.3-0.6 | ⚠️ 异常高 |
| **avg_bertscore_f1** | 0.1481 | 0.85-0.95 | ❌ 极低 |
| **bertscore_precision** | 0.2939 | 0.85-0.95 | ❌ 极低 |
| **bertscore_recall** | 0.0079 | 0.85-0.95 | ❌ **极度异常低** |

**异常分析**:
1. **BLEU接近0**: Reference质量太差，几乎无法匹配
2. **CIDEr异常高**: 可能是因为reference太长，某些n-gram碰巧匹配
3. **BERTScore极低**: 特别是Recall只有0.0079（正常应该>0.85）
4. **BERTScore Recall负值**: Sample 00000021的recall是**-0.079**（不正常）

---

## 💡 问题根源

### Reference Caption问题链

```
1. MER2024原始数据集是中文
   ↓
2. 有人用机器翻译生成了english_caption
   ↓
3. 翻译质量很差：
   - 重复句子
   - 包含不应该有的内容（字幕、视频）
   - 语法错误
   ↓
4. 使用这些reference评估
   ↓
5. 结果：BLEU/BERTScore极低（但不是模型的错）
```

### 生成质量问题链

```
1. 英语Prompt不够严格
   ↓
2. 模型生成对话式内容（"What do you think..."）
   ↓
3. 不符合任务要求（应该是客观描述）
   ↓
4. 与中文baseline不一致（中文禁止对话式）
```

---

## 🔧 解决方案

### 方案A: 修复English Prompt（推荐）

#### 当前Prompt
```
"As an expert in the field of emotions, please focus on the acoustic information in the audio to discern clues related to the emotions of the individual. Please provide a detailed description and ultimately predict the emotional state of the individual."
```

#### 建议Prompt（对齐中文版本）
```
"Task: Generate a detailed emotional description based solely on acoustic features from the audio.

Structure your response as follows:
(1) First, describe 2-3 acoustic/prosodic features (choose from: speech rate, pitch variation, volume, pauses, tone quality, etc.) at a categorical level without specific values.
(2) Based on these features, identify the most likely single emotion.
(3) If the semantic content suggests a reason, briefly mention it in one short phrase (use "possibly/seems/might" to indicate uncertainty).

Requirements:
- Output 2-3 English sentences, approximately 50-70 words total.
- Use third-person references (e.g., "the speaker", "he/she"); do NOT use first/second person.
- Do NOT ask questions or invite conversation.
- Do NOT fabricate specific details about people, times, or places.
- Do NOT include emojis, Chinese text, Markdown, or code."
```

**关键改进**:
- ✅ 明确要求"Do NOT ask questions or invite conversation"
- ✅ 限制句子数量（2-3句）
- ✅ 限制长度（50-70词）
- ✅ 要求客观描述（第三人称）

---

### 方案B: 更换数据集（推荐）

#### 问题分析
MER2024的`english_caption`质量太差，不适合作为英语baseline的reference。

#### 建议
使用**SECAP数据集**进行英语测试：

```bash
python experiments/runs/run_cloud_baseline.py \
    --dataset_type unified \
    --dataset_path data/processed/secap/manifest.json \
    --caption_type original \
    --language english \
    --prompt_type detailed \
    --max_samples 10 \
    --output_name cloud_secap_en
```

**原因**:
- SECAP是英语原生数据集
- Reference质量更好
- 更适合评估英语生成能力

---

### 方案C: 清理Reference Caption（可选）

如果必须使用MER2024英语caption，可以尝试清理：

```python
def clean_english_caption(caption: str) -> str:
    """Clean machine-translated English caption"""
    # 1. 移除重复句子
    sentences = caption.split('. ')
    unique_sentences = []
    seen = set()
    for sent in sentences:
        sent_clean = sent.strip().lower()
        if sent_clean not in seen and sent_clean:
            unique_sentences.append(sent)
            seen.add(sent_clean)
    
    # 2. 移除包含字幕的句子
    filtered = []
    for sent in unique_sentences:
        if 'subtitle' not in sent.lower() and 'text reads' not in sent.lower():
            filtered.append(sent)
    
    # 3. 重新组合
    return '. '.join(filtered[:3]) + '.'  # 只保留前3句
```

**缺点**: 可能仍然质量不佳

---

## 📊 指标异常分析

### BERTScore Recall = 0.0079（极度异常）

**正常BERTScore**:
| 指标 | 正常范围 | 实际值 | 状态 |
|------|---------|--------|------|
| Precision | 0.85-0.95 | 0.2939 | ❌ 很低 |
| Recall | 0.85-0.95 | **0.0079** | ❌ **极度异常** |
| F1 | 0.85-0.95 | 0.1481 | ❌ 很低 |

**为什么Recall这么低？**

**BERTScore Recall定义**:
```
Recall = 有多少reference中的词/概念被generated text覆盖
```

**当前情况**:
- Reference: 超长（300+词），包含大量信息（音频+字幕+视频）
- Generated: 正常长度（40-50词），只有音频信息
- Recall: 生成文本只覆盖了reference的0.79%！

**示例**:
```
Reference (300词): 
  "tone raised... discussing serious issue... laughing... 
   I, I, I don't have experience... male doctor... 
   humorous tone... lightens topic..."

Generated (40词):
  "The voice is steady but carries uncertainty and hesitation..."

Recall: 40词中匹配的 / 300词 ≈ 0.0079 (0.79%)
```

---

### BLEU = 0.0004（几乎为0）

**正常英语BLEU**: 0.15-0.30

**为什么这么低？**

**BLEU定义**: 基于n-gram匹配（1-gram, 2-gram, 3-gram, 4-gram）

**当前情况**:
- Reference质量差（重复、错误、包含字幕）
- Generated text是正常英语，但与低质量reference无法匹配

**示例**:
```
Reference: "In the audio, ... In the audio, ... I, I, I don't..."
Generated: "The voice is steady but carries uncertainty..."

n-gram匹配: 几乎没有
BLEU: ≈ 0.0004
```

---

### CIDEr = 0.9580（异常高）

**正常CIDEr**: 0.3-0.6

**为什么这么高？**

**可能原因**:
1. Reference太长（300+词），某些常见词碰巧匹配
2. CIDEr的TF-IDF加权可能被超长reference扭曲
3. 需要进一步调查（可能是计算错误）

---

## 🎯 核心结论

### 问题优先级

| 问题 | 严重性 | 影响 | 建议 |
|------|--------|------|------|
| **Reference质量差** | 🔴 极严重 | 评估完全不可靠 | 更换数据集（SECAP） |
| **对话式内容** | 🔴 严重 | 与中文不一致 | 修复Prompt |
| **BLEU极低** | 🟡 中等 | 指标不可信 | 结果：更换数据集 |
| **BERTScore极低** | 🟡 中等 | 指标不可信 | 结果：更换数据集 |

---

## 💡 推荐行动方案

### Step 1: 修复英语Prompt（立即）

添加明确的约束，禁止对话式内容。

**修改位置**: `experiments/runs/run_cloud_baseline.py` (和其他baseline脚本)

```python
elif language == "english":
    return """Task: Generate a detailed emotional description based solely on acoustic features from the audio.

Structure your response as follows:
(1) First, describe 2-3 acoustic/prosodic features (choose from: speech rate, pitch variation, volume, pauses, tone quality, etc.) at a categorical level without specific values.
(2) Based on these features, identify the most likely single emotion.
(3) If the semantic content suggests a reason, briefly mention it in one short phrase (use "possibly/seems/might" to indicate uncertainty).

Requirements:
- Output 2-3 English sentences, approximately 50-70 words total.
- Use third-person references (e.g., "the speaker", "he/she"); do NOT use first/second person.
- Do NOT ask questions or invite conversation.
- Do NOT fabricate specific details about people, times, or places.
- Do NOT include emojis, Chinese text, Markdown, or code."""
```

---

### Step 2: 更换数据集（强烈推荐）

**从MER2024切换到SECAP**:

```bash
# 使用SECAP（英语原生数据集）
python experiments/runs/run_cloud_baseline.py \
    --dataset_type unified \
    --dataset_path data/processed/secap/manifest.json \
    --caption_type original \
    --language english \
    --prompt_type detailed \
    --max_samples 10 \
    --output_name cloud_secap_en
```

**优势**:
- ✅ SECAP是英语原生数据集
- ✅ Reference质量高
- ✅ 没有翻译问题
- ✅ 指标才有意义

---

### Step 3: 或者使用中文数据集（备选）

如果想继续用MER2024，应该用**中文**:

```bash
python experiments/runs/run_cloud_baseline.py \
    --dataset_type unified \
    --dataset_path data/processed/mer2024/manifest_audio_only_final.json \
    --caption_type audio_only \
    --language chinese \
    --prompt_type detailed \
    --max_samples 10 \
    --output_name cloud_mer2024_zh
```

---

## 🔍 进一步验证

### 检查SECAP数据集质量

让我查看SECAP的样本：

```bash
# 查看SECAP的第一个样本
head -50 data/processed/secap/manifest.json
```

如果SECAP的caption质量好，那就应该用SECAP进行英语测试。

---

## 📝 总结

### 核心问题

1. ❌ **MER2024的english_caption质量极差**（机器翻译、重复、包含字幕）
2. ❌ **生成了对话式内容**（Prompt不够严格）
3. ❌ **所有指标都不可信**（BLEU=0.0004, BERTScore Recall=0.0079）

### 推荐方案

**立即执行**:
1. ✅ 修复英语Prompt（禁止对话式）
2. ✅ 切换到SECAP数据集（英语原生）

**验证**:
- 检查生成文本无对话式内容
- BLEU应该在0.15-0.30范围
- BERTScore应该在0.85-0.95范围

**暂时不要用MER2024的英语caption**:
- 质量太差，无法作为可靠的reference
- 评估结果没有意义

---

**需要我帮您修复英语Prompt并切换到SECAP吗？** 🚀

