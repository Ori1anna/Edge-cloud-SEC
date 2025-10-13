# OV-MER 任务详细分析与实施计划

## 📋 任务概述

您需要参加 **MER2024 Track 3: OV-MER (Open-Vocabulary Multimodal Emotion Recognition)** 比赛。

### 核心目标
从您已生成的**英文情感描述**中抽取**开放词汇情感标签列表**，并使用官方评测脚本计算得分。

---

## 🎯 任务流程（完整链路）

```
Step 1: 音频输入
   ↓
Step 2: Qwen-2.5-Omni-7B 生成英文情感描述
   ↓ (您已完成，输出在 .json 的 "generated_text" 字段)
Step 3: 从描述中抽取情感标签列表 (待完成)
   ↓
Step 4: 保存为 predict-openset.csv (待完成)
   ↓
Step 5: 使用官方评测脚本计算得分 (待完成)
```

---

## 📊 当前状态分析

### ✅ 已完成部分

1. **音频情感描述生成**
   - 您已使用 Qwen-2.5-Omni-7B 生成了英文情感描述
   - 输出文件：`experiments/results/cloud_mer_en_test1.json`
   - 格式：每个样本有 `generated_text` 字段

   **示例**（来自您的输出）:
   ```json
   {
     "file_id": "sample_00000000",
     "generated_text": "The voice is steady but carries a tone of uncertainty and hesitation. It sounds like the speaker is trying to reassure someone or explain something without much confidence. What do you think might be causing this emotion?"
   }
   ```

2. **评测环境准备**
   - ✅ 已克隆 MERTools 仓库：`/data/gpfs/projects/punim2341/jiajunlu/edgecloud-sec/MERTools/MER2024`
   - ✅ 已有官方真值文件：`ov_store/check-openset.csv`
   - ✅ 已有同义词缓存：`ov_store/openset-synonym.zip`（用于离线评测）
   - ✅ 已有评测脚本：`main-ov.py`

---

### ⏳ 待完成部分

#### **Step 3: 情感标签抽取**（核心任务）

**输入**: 英文情感描述（generated_text）
```
"The voice is steady but carries a tone of uncertainty and hesitation..."
```

**输出**: 情感标签列表（1-8个，去重，小写）
```python
['uncertainty', 'hesitation', 'nervousness']
```

**方法**: 使用 Qwen-2.5-Omni-7B + 特定提示词（来自论文 Appendix #5）

---

#### **Step 4: 保存为 CSV**

**格式要求**（参考 `ov_store/predict-openset.csv`）:
```csv
name,openset
sample_00000000,"['uncertainty', 'hesitation', 'nervousness']"
sample_00000007,"['happy', 'excited']"
```

**关键点**:
- 列名必须是 `name,openset`
- `name` 必须与 `check-openset.csv` 中的样本名一致
- `openset` 是**字符串形式的Python列表**（注意引号）

---

#### **Step 5: 评测得分**

运行官方评测脚本，计算 **set-level accuracy** 和 **recall**。

---

## 📁 文件结构分析

### 官方评测资产

```
MERTools/MER2024/ov_store/
├── check-openset.csv          # 官方真值（英文标签）
├── predict-openset.csv         # 预测结果模板
└── openset-synonym.zip         # 同义词缓存（离线评测用）
```

#### `check-openset.csv` 格式（真值）

```csv
name,openset
sample_00000000,"['pleasurable', 'self-deprecating', 'relaxed', 'complaining', 'relieved', 'humorous', 'joyful', 'happy']"
sample_00000007,"['excited', 'joyful', 'happy']"
sample_00000021,"['suspect', 'question', 'doubt']"
```

**特点**:
- 每个样本有 1-8 个情感标签
- 标签是英文单词或短语（小写）
- 多样性很高（开放词汇）

---

#### `predict-openset.csv` 格式（您的预测）

```csv
name,openset
sample_00000000,"['confused', 'helpless']"
sample_00000007,"['happy', 'satisfied']"
```

**要求**:
- 必须包含所有测试样本（334个样本）
- `name` 必须与真值文件一致
- `openset` 格式：`"['label1', 'label2', ...]"`

---

### 评测脚本分析

#### `main-ov.py` 核心函数

**函数1: `calculate_openset_overlap_rate_mer2024`**（离线评测）

```python
def calculate_openset_overlap_rate_mer2024(gt_csv, pred_csv, synonym_root):
    # 1. 读取真值和预测
    # 2. 读取同义词映射（从 .npy 文件）
    # 3. 将标签映射到同义词组
    # 4. 计算集合指标
    #    - accuracy = |GT ∩ Pred| / |Pred|  (预测的准确性)
    #    - recall   = |GT ∩ Pred| / |GT|    (真值的覆盖率)
    # 5. 返回平均 accuracy 和 recall
```

**关键逻辑**:
- 使用**同义词映射**将不同表达归为同一组
  - 例如: `['angry', 'mad', 'furious']` → 都映射到 `'angry'`
- 计算**集合交集**（不考虑顺序）
- 如果 `Pred` 为空，accuracy 和 recall 都记为 0

**评测指标**:
```
Set-level Accuracy = |GT ∩ Pred| / |Pred|
Set-level Recall   = |GT ∩ Pred| / |GT|
Final Score        = (Accuracy + Recall) / 2
```

**示例**:
```python
GT   = {'angry', 'disappointed', 'frustrated'}
Pred = {'angry', 'sad', 'upset'}

# 映射后（假设 'upset' 和 'frustrated' 是同义词）
GT_mapped   = {'angry', 'disappointed', 'frustrated'}
Pred_mapped = {'angry', 'sad', 'frustrated'}

# 计算
Intersection = {'angry', 'frustrated'}  # 2个
Accuracy = 2 / 3 = 0.667
Recall   = 2 / 3 = 0.667
```

---

**函数2: `generate_openset_synonym_mer2024`**（在线生成同义词）

```python
def generate_openset_synonym_mer2024(gt_csv, pred_csv, synonym_root, gptmodel):
    # 使用 GPT-3.5 在线生成同义词映射
    # 需要 OpenAI API key
    # 会产生成本
```

**注意**: 您**不需要**使用这个函数，因为官方已提供 `openset-synonym.zip`。

---

## 🔧 技术实现方案

### 方案概述

创建一个 Python 脚本 `tools/extract_emotion_labels.py`，用于：
1. 读取您的英文情感描述（.json 文件）
2. 调用 Qwen-2.5-Omni-7B 抽取情感标签
3. 保存为 `predict-openset.csv`

---

### 提示词设计（基于论文 Appendix #5）

#### 论文原始提示词（GPT-3.5）

```
Please assume the role of an expert in the field of emotions. 
We provide clues that may be related to the emotions of the characters. 
Based on the provided clues, please identify the emotional states of the main characters. 
Please separate different emotional categories with commas and output only the clearly identifiable emotional categories in a list format. 
If none are identified, please output an empty list.
```

#### 优化后的提示词（适配 Qwen + JSON 输出）

```
You are an expert in emotion recognition. 

Given the following emotional description from audio analysis, extract a list of emotion labels (1-8 labels) that best represent the emotional states described.

Requirements:
- Output ONLY a JSON array of emotion labels
- Each label should be a single English word or short phrase (lowercase)
- Remove duplicates
- If no clear emotion is identified, output an empty list []
- Do NOT include explanations or additional text

Examples:
Input: "The speaker sounds angry and frustrated, with a loud and aggressive tone."
Output: ["angry", "frustrated", "aggressive"]

Input: "The voice is calm and steady, conveying confidence and determination."
Output: ["calm", "confident", "determined"]

Now, extract emotion labels from this description:
"{description}"

Output (JSON array only):
```

---

### 脚本设计

#### 输入参数

```bash
python tools/extract_emotion_labels.py \
    --input_json experiments/results/cloud_mer_en_test1.json \
    --output_csv MERTools/MER2024/ov_store/predict-openset.csv \
    --model_name Qwen/Qwen2.5-Omni-7B \
    --device cuda:0
```

#### 核心逻辑

```python
# 伪代码
def extract_emotion_labels(description: str, model, processor) -> List[str]:
    """
    从情感描述中抽取标签列表
    
    Args:
        description: 英文情感描述
        model: Qwen模型
        processor: Qwen处理器
    
    Returns:
        情感标签列表（小写，去重）
    """
    # 1. 构建提示词
    prompt = EMOTION_EXTRACTION_PROMPT.format(description=description)
    
    # 2. 调用模型生成
    response = model.generate(prompt)
    
    # 3. 解析 JSON 输出
    try:
        labels = json.loads(response)
        # 清理：小写、去重、限制1-8个
        labels = [label.lower().strip() for label in labels]
        labels = list(dict.fromkeys(labels))  # 去重保持顺序
        labels = labels[:8]  # 最多8个
        return labels
    except:
        return []  # 解析失败返回空列表


def main(input_json, output_csv):
    # 1. 加载模型
    model, processor = load_qwen_model()
    
    # 2. 读取输入 JSON
    with open(input_json) as f:
        data = json.load(f)
    
    # 3. 处理每个样本
    results = []
    for sample in data['detailed_results']:
        file_id = sample['file_id']
        description = sample['generated_text']
        
        # 抽取标签
        labels = extract_emotion_labels(description, model, processor)
        
        results.append({
            'name': file_id,
            'openset': str(labels)  # 转为字符串形式的列表
        })
    
    # 4. 保存为 CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
```

---

### 输出格式处理

**关键点**: `openset` 列必须是**字符串形式的Python列表**

```python
# ✅ 正确格式
"['angry', 'frustrated']"

# ❌ 错误格式
['angry', 'frustrated']  # 这是真的列表，CSV会出错
"angry, frustrated"      # 这是逗号分隔字符串
```

**实现**:
```python
import pandas as pd

results = []
for sample in samples:
    labels = ['angry', 'frustrated']  # 列表
    results.append({
        'name': sample['file_id'],
        'openset': str(labels)  # 转为字符串 "['angry', 'frustrated']"
    })

df = pd.DataFrame(results)
df.to_csv(output_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)
```

---

## 🎯 评测流程

### Step 1: 解压同义词缓存

```bash
cd /data/gpfs/projects/punim2341/jiajunlu/edgecloud-sec/MERTools/MER2024/ov_store
unzip openset-synonym.zip
# 会得到 openset-synonym/ 目录，里面有 334 个 .npy 文件
```

**验证**:
```bash
ls openset-synonym/ | wc -l  # 应该是 334
```

---

### Step 2: 运行评测脚本

```bash
cd /data/gpfs/projects/punim2341/jiajunlu/edgecloud-sec/MERTools/MER2024

python main-ov.py calculate_openset_overlap_rate_mer2024 \
    --gt_csv='ov_store/check-openset.csv' \
    --pred_csv='ov_store/predict-openset.csv' \
    --synonym_root='ov_store/openset-synonym'
```

**预期输出**:
```
process number (after filter): 334
set level accuracy: 0.5818
recall: 0.4978
avg score: 0.5398
```

---

### Step 3: 理解评测结果

| 指标 | 含义 | 计算公式 |
|------|------|---------|
| **Accuracy** | 预测标签的准确性 | \|GT ∩ Pred\| / \|Pred\| |
| **Recall** | 真值标签的覆盖率 | \|GT ∩ Pred\| / \|GT\| |
| **Avg Score** | 综合得分 | (Accuracy + Recall) / 2 |

**示例解释**:
```
GT   = ['angry', 'disappointed', 'frustrated']  (3个真值标签)
Pred = ['angry', 'sad']                         (2个预测标签)

Intersection = ['angry']  (1个匹配)

Accuracy = 1/2 = 0.5    (预测的2个中，1个正确)
Recall   = 1/3 = 0.333  (真值的3个中，1个被覆盖)
```

**优化方向**:
- **提高 Accuracy**: 减少错误预测（提高精度）
- **提高 Recall**: 增加正确预测（提高召回）
- **平衡**: 预测 3-5 个标签通常效果较好

---

## 🚨 关键注意事项

### 1. 样本名称必须对齐

**check-openset.csv** 中的样本名:
```
sample_00000000
sample_00000007
sample_00000021
...
```

**您的 JSON** 中的 `file_id`:
```json
"file_id": "sample_00000000"
```

**要求**: 必须完全一致！

**验证**:
```python
# 检查是否所有样本都对齐
gt_names = set(pd.read_csv('check-openset.csv')['name'])
pred_names = set(pd.read_csv('predict-openset.csv')['name'])

missing = gt_names - pred_names
extra = pred_names - gt_names

print(f"Missing: {len(missing)}")  # 应该是 0
print(f"Extra: {len(extra)}")      # 应该是 0
```

---

### 2. 标签格式要求

**✅ 正确格式**:
```python
['angry', 'frustrated']           # 小写
['happy', 'joyful', 'excited']    # 多个标签
['fear']                          # 单个标签
[]                                # 空列表（无法识别时）
```

**❌ 错误格式**:
```python
['Angry', 'Frustrated']           # 大写（会被转为小写）
['angry', 'angry']                # 重复（应去重）
['angry', 'frustrated', ...]      # 超过8个（应截断）
```

---

### 3. 同义词处理

官方的 `openset-synonym.zip` 已经预处理了同义词映射。

**示例**（来自 .npy 文件）:
```python
# sample_00000000.npy 内容（假设）
[
    ['pleasurable', 'pleasant'],
    ['self-deprecating', 'self-deprecation'],
    ['relaxed', 'calm', 'ease'],
    ['complaining', 'complaint'],
    ...
]
```

**含义**: 同一个子列表中的词会被视为同义词，映射到第一个词。

**您的任务**: 不需要手动处理同义词，评测脚本会自动处理。

---

### 4. 对话式内容问题

**回顾您的英文生成问题**（之前发现的）:
```
"...What do you think might be causing this emotion?"  ❌
```

**影响**: 对话式内容可能导致标签抽取不准确。

**解决方案**:
1. **短期**: 在标签抽取时，提示词明确要求"只输出情感标签"
2. **长期**: 修复英文 prompt，重新生成描述（推荐）

---

## 📈 预期性能分析

### 官方 Baseline（使用 predict-openset.csv 模板）

```
Accuracy: 0.5818
Recall:   0.4978
Avg:      0.5398
```

**分析**:
- 这是官方提供的一个示例预测结果
- 您的目标应该是**超过这个基线**

---

### 您的预期性能

**影响因素**:

1. **情感描述质量**（最关键）
   - ✅ 如果描述准确、客观 → 标签抽取容易
   - ❌ 如果有对话式内容 → 标签抽取困难

2. **标签抽取准确性**
   - 提示词设计
   - 模型理解能力

3. **标签数量**
   - 太少（1-2个）→ Recall 低
   - 太多（6-8个）→ Accuracy 低
   - **最佳**: 3-5个标签

---

### 优化策略

| 阶段 | 策略 | 预期提升 |
|------|------|---------|
| **Phase 1** | 使用当前描述 + 优化提示词 | Baseline → 0.55-0.60 |
| **Phase 2** | 修复英文 prompt，重新生成描述 | 0.60 → 0.65-0.70 |
| **Phase 3** | 微调标签数量（3-5个） | 0.70 → 0.75+ |

---

## 🛠️ 实施步骤（详细清单）

### ✅ Step 1: 环境检查

```bash
# 1.1 检查 MERTools 仓库
cd /data/gpfs/projects/punim2341/jiajunlu/edgecloud-sec/MERTools/MER2024
ls ov_store/
# 应该看到: check-openset.csv, predict-openset.csv, openset-synonym.zip

# 1.2 解压同义词缓存
cd ov_store
unzip -q openset-synonym.zip
ls openset-synonym/ | wc -l  # 应该是 334

# 1.3 检查评测脚本
cd ..
python main-ov.py --help  # 应该能正常运行
```

---

### ⏳ Step 2: 创建标签抽取脚本

**文件**: `tools/extract_emotion_labels.py`

**功能**:
1. 读取英文情感描述 JSON
2. 使用 Qwen 抽取情感标签
3. 保存为 predict-openset.csv

**关键点**:
- 提示词设计（基于论文 #5）
- JSON 解析（处理模型输出）
- CSV 格式（字符串形式的列表）

---

### ⏳ Step 3: 运行标签抽取

```bash
python tools/extract_emotion_labels.py \
    --input_json experiments/results/cloud_mer_en_test1.json \
    --output_csv MERTools/MER2024/ov_store/predict-openset-qwen.csv \
    --model_name Qwen/Qwen2.5-Omni-7B \
    --device cuda:0
```

**预期输出**:
```
Processing 10 samples...
Sample 1/10: sample_00000000 -> ['uncertainty', 'hesitation']
Sample 2/10: sample_00000007 -> ['happy', 'cheerful']
...
Saved to predict-openset-qwen.csv
```

---

### ⏳ Step 4: 验证输出格式

```bash
# 检查 CSV 格式
head -5 MERTools/MER2024/ov_store/predict-openset-qwen.csv

# 检查样本数量
wc -l MERTools/MER2024/ov_store/predict-openset-qwen.csv
# 应该是 11 行（10个样本 + 1个表头）

# 检查样本名称对齐
python -c "
import pandas as pd
gt = pd.read_csv('MERTools/MER2024/ov_store/check-openset.csv')
pred = pd.read_csv('MERTools/MER2024/ov_store/predict-openset-qwen.csv')
print('GT samples:', len(gt))
print('Pred samples:', len(pred))
print('Missing:', set(gt['name']) - set(pred['name']))
"
```

---

### ⏳ Step 5: 运行评测

```bash
cd MERTools/MER2024

python main-ov.py calculate_openset_overlap_rate_mer2024 \
    --gt_csv='ov_store/check-openset.csv' \
    --pred_csv='ov_store/predict-openset-qwen.csv' \
    --synonym_root='ov_store/openset-synonym'
```

**预期输出**:
```
process number (after filter): 10
set level accuracy: 0.XXXX
recall: 0.XXXX
avg score: 0.XXXX
```

---

### ⏳ Step 6: 分析结果

**如果得分低于预期**:

1. **检查标签抽取质量**
   ```bash
   # 查看前几个样本的抽取结果
   head -10 ov_store/predict-openset-qwen.csv
   ```

2. **对比真值**
   ```bash
   # 查看同一样本的真值和预测
   python -c "
   import pandas as pd
   gt = pd.read_csv('ov_store/check-openset.csv')
   pred = pd.read_csv('ov_store/predict-openset-qwen.csv')
   
   sample = 'sample_00000000'
   print('GT:', gt[gt['name']==sample]['openset'].values[0])
   print('Pred:', pred[pred['name']==sample]['openset'].values[0])
   "
   ```

3. **调整提示词**
   - 如果标签太少 → 提示"identify 3-5 emotions"
   - 如果标签太多 → 提示"only the most prominent emotions"
   - 如果标签不准 → 增加示例（few-shot）

---

## 📊 完整数据集评测（最终提交）

### 当前测试（10个样本）

您的 `cloud_mer_en_test1.json` 只有 10 个样本（`--max_samples 10`）。

---

### 完整评测（334个样本）

**Step 1**: 生成完整数据集的英文描述

```bash
# 使用完整的 MER2024 测试集
python experiments/runs/run_cloud_baseline.py \
    --dataset_type unified \
    --dataset_path data/processed/mer2024/manifest_audio_only_final.json \
    --caption_type audio_only \
    --language english \
    --prompt_type detailed \
    --output_name cloud_mer2024_en_full
# 注意：不加 --max_samples，会处理所有样本
```

**Step 2**: 抽取标签

```bash
python tools/extract_emotion_labels.py \
    --input_json experiments/results/cloud_mer2024_en_full.json \
    --output_csv MERTools/MER2024/ov_store/predict-openset-full.csv \
    --model_name Qwen/Qwen2.5-Omni-7B \
    --device cuda:0
```

**Step 3**: 评测

```bash
cd MERTools/MER2024
python main-ov.py calculate_openset_overlap_rate_mer2024 \
    --gt_csv='ov_store/check-openset.csv' \
    --pred_csv='ov_store/predict-openset-full.csv' \
    --synonym_root='ov_store/openset-synonym'
```

---

## 🎓 技术细节补充

### 1. 为什么使用 Qwen 而不是 GPT？

**论文使用 GPT-3.5**:
- 论文 Appendix #5 使用 GPT-3.5-turbo 抽取标签
- 但这只是一个**基线方法**

**您使用 Qwen 的优势**:
- ✅ **本地部署**：无 API 成本
- ✅ **完全可控**：可以调整提示词、温度等
- ✅ **可复现**：不受 OpenAI API 变化影响
- ✅ **多模态**：Qwen-2.5-Omni 支持音频输入（虽然这里只用文本）

**合理性**: 官方允许使用任何方法抽取标签，只要最终格式正确。

---

### 2. 同义词映射原理

**官方提供的 `openset-synonym.zip`**:
- 每个样本一个 `.npy` 文件
- 文件名：`sample_XXXXXXXX.npy`
- 内容：同义词簇列表

**示例**（假设 `sample_00000000.npy`）:
```python
import numpy as np
data = np.load('sample_00000000.npy', allow_pickle=True)
print(data)
# 输出:
# [
#     ['pleasurable', 'pleasant', 'enjoyable'],
#     ['angry', 'mad', 'furious'],
#     ['relaxed', 'calm', 'ease'],
#     ...
# ]
```

**映射逻辑**:
```python
# 构建映射字典
synonym_map = {}
for group in data:
    canonical = group[0]  # 第一个词作为代表
    for word in group:
        synonym_map[word] = canonical

# 例如:
# synonym_map['pleasant'] = 'pleasurable'
# synonym_map['enjoyable'] = 'pleasurable'
# synonym_map['mad'] = 'angry'
# synonym_map['furious'] = 'angry'
```

**评测时应用**:
```python
# GT 和 Pred 都映射到代表词
gt_mapped = {synonym_map.get(w, w) for w in gt}
pred_mapped = {synonym_map.get(w, w) for w in pred}

# 然后计算交集
intersection = gt_mapped & pred_mapped
```

---

### 3. 为什么需要离线同义词缓存？

**问题**: 情感词汇有大量同义词
- `happy` = `joyful` = `pleased` = `delighted`
- `angry` = `mad` = `furious` = `enraged`

**如果不处理同义词**:
```python
GT   = ['happy', 'excited']
Pred = ['joyful', 'excited']  # 'joyful' 和 'happy' 是同义词

# 不处理同义词
Intersection = {'excited'}  # 只有1个匹配
Accuracy = 1/2 = 0.5

# 处理同义词
Intersection = {'happy', 'excited'}  # 2个匹配
Accuracy = 2/2 = 1.0
```

**官方方案**:
- 使用 GPT-3.5 生成同义词簇（在线，需 API）
- 或使用预生成的缓存（离线，免费）

**您应该使用**: 预生成的缓存（`openset-synonym.zip`）

---

## 🚀 下一步行动

### 立即执行（优先级排序）

#### **Priority 1: 解压同义词缓存**（1分钟）

```bash
cd /data/gpfs/projects/punim2341/jiajunlu/edgecloud-sec/MERTools/MER2024/ov_store
unzip openset-synonym.zip
ls openset-synonym/ | wc -l  # 验证是 334
```

---

#### **Priority 2: 创建标签抽取脚本**（30分钟）

创建 `tools/extract_emotion_labels.py`，实现：
1. 读取 JSON（`generated_text` 字段）
2. 调用 Qwen 抽取标签（使用优化的提示词）
3. 保存为 CSV（正确格式）

**我可以帮您编写这个脚本！**

---

#### **Priority 3: 测试评测流程**（10分钟）

使用您现有的 10 个样本测试：
1. 运行标签抽取
2. 运行评测脚本
3. 分析结果

---

#### **Priority 4: 修复英文 Prompt**（可选，但推荐）

修复之前发现的对话式内容问题，重新生成高质量的英文描述。

---

#### **Priority 5: 完整数据集评测**（最终）

处理所有 334 个样本，获得最终得分。

---

## 📝 总结

### 核心任务链路

```
英文情感描述 (generated_text)
   ↓ [Qwen + 提示词]
情感标签列表 (['angry', 'frustrated'])
   ↓ [保存为 CSV]
predict-openset.csv
   ↓ [官方评测脚本 + 同义词缓存]
Set-level Accuracy & Recall
```

---

### 关键文件

| 文件 | 作用 | 状态 |
|------|------|------|
| `cloud_mer_en_test1.json` | 英文情感描述（10个样本） | ✅ 已有 |
| `check-openset.csv` | 官方真值（334个样本） | ✅ 已有 |
| `openset-synonym.zip` | 同义词缓存 | ✅ 已有（需解压） |
| `main-ov.py` | 评测脚本 | ✅ 已有 |
| `extract_emotion_labels.py` | 标签抽取脚本 | ⏳ 待创建 |
| `predict-openset-qwen.csv` | 您的预测结果 | ⏳ 待生成 |

---

### 预期时间线

| 任务 | 预计时间 | 累计时间 |
|------|---------|---------|
| 解压同义词缓存 | 1分钟 | 1分钟 |
| 创建抽取脚本 | 30分钟 | 31分钟 |
| 测试10个样本 | 10分钟 | 41分钟 |
| 修复英文prompt（可选） | 20分钟 | 61分钟 |
| 完整数据集评测 | 1-2小时 | 3小时 |

---

### 成功标准

- ✅ 评测脚本能正常运行
- ✅ 得分 > 0.54（超过官方baseline）
- ✅ 格式完全符合提交要求
- 🎯 目标得分 > 0.65（优秀）

---

## 🤝 我可以帮您做什么？

1. **✅ 立即创建 `extract_emotion_labels.py` 脚本**
2. **✅ 优化提示词设计**
3. **✅ 测试评测流程**
4. **✅ 分析结果并优化**

**您希望我从哪一步开始？** 🚀

