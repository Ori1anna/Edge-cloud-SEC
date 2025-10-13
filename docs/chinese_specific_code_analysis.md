# 中文特定代码分析 - 支持英语所需的修改

## 📋 分析目标

检查所有代码中专用于中文的部分，以便支持英语生成。

---

## 🔍 中文特定代码位置

### 1. **Speculative Decoding 生成逻辑** (`src/speculative_decoding.py`)

#### 1.1 CJK字符检测函数

**位置**: 第929-935行, 961-967行

**代码**:
```python
def _is_cjk(token_id):
    """Check if token contains Chinese/Japanese/Korean characters"""
    try:
        s = tokenizer.decode([token_id], skip_special_tokens=True)
        return any('\u4e00' <= ch <= '\u9fff' for ch in s)  # ✅ CJK范围
    except:
        return False
```

**影响的逻辑**：
1. **Repetition penalty** (第942-945行)：仅对CJK token应用1.22的惩罚
2. **Same-char blocking** (第984-986行)：仅阻止CJK字符的immediate重复
3. **Content-only trigram** (第992-1004行)：仅当recent window全是CJK时应用

**问题**：
- ❌ 对英语token会返回False
- ❌ 英语不会应用repetition penalty、same-char blocking、trigram ban

---

#### 1.2 中文标点符号

**位置**: 第979, 1009-1010行

**代码**:
```python
# 第979行
PUNCT_IDS = _ids_for(['，', '。', '、', '：', ':', '；', '！', '？'])
                    # ↑ 只有中文标点

# 第1009-1010行
COMMA_LIKE = _ids_for(['，', '、', '：', ':'])  # 中文逗号类
PERIOD_LIKE = _ids_for(['。'])  # 中文句号
```

**问题**：
- ❌ **没有英文标点**：`,`, `.`, `;`, `!`, `?`
- ❌ 英语生成时，标点闸门无效

---

#### 1.3 标点闸门（Punctuation Gate）

**位置**: 第1012-1038行

**代码**:
```python
# 统计自上次标点以来的CJK字符数
since_punct = 0
for t in reversed(hist):
    if t in PUNCT_IDS:
        break
    s = tokenizer.decode([t])
    if any('\u4e00' <= ch <= '\u9fff' for ch in s):  # ❌ 只统计CJK
        since_punct += 1

# 逗号闸门：要求至少4个CJK字符
if since_punct < 4:
    logits[COMMA_LIKE] = -inf

# 句号闸门：要求至少5个CJK字符  
if since_punct < 5:
    logits[PERIOD_LIKE] -= 3.5
```

**问题**：
- ❌ 只统计CJK字符，不统计英语单词/token
- ❌ 阈值（4字，5字）是针对中文设计的
- ❌ 英语生成时基本不起作用

---

#### 1.4 句子结束判断

**位置**: 第158-160行

**代码**:
```python
sentence_endings = ['。', '.']  # ✅ 包含中英文
return token_text in sentence_endings
```

**状态**: ✅ **已支持英语**

---

### 2. **Stopping Criteria** (`src/models/stopping_criteria.py`)

**位置**: 第17, 134行

**代码**:
```python
# 第17行
sentence_end_chars=("。", ".")  # ✅ 默认包含中英文

# 第134行
sentence_end_chars=("。", ".")  # ✅ 默认包含中英文
```

**状态**: ✅ **已支持英语**

---

### 3. **BLEU计算** (所有baseline脚本)

**位置**: 
- `run_edge_baseline_cpu_limited.py` 第535行
- `run_cloud_baseline.py` 第265行
- `run_speculative_decoding_cpu_limited.py` 第573行

**代码**:
```python
# 第535行（edge baseline）
corpus_bleu = sacrebleu.corpus_bleu(hyps, refs, tokenize='zh')  # ❌ 中文tokenization
overall_bleu = corpus_bleu.score / 100.0

# 结果key
"corpus_bleu_zh": overall_bleu  # ❌ 固定使用"_zh"
```

**问题**：
- ❌ **硬编码使用中文tokenization** (`tokenize='zh'`)
- ❌ 英语应该使用 `tokenize='13a'` 或 `tokenize='intl'`
- ❌ 结果key名称固定为 `corpus_bleu_zh`

---

### 4. **Prompt模板** (所有baseline脚本)

**位置**: 
- `run_edge_baseline_cpu_limited.py` 第316-350行
- `run_cloud_baseline.py` 第61-114行
- `run_speculative_decoding_cpu_limited.py` 第370-421行

**代码示例** (`run_cloud_baseline.py`):
```python
def get_prompt_template(prompt_type: str, language: str) -> str:
    """Get prompt template based on prompt type and language"""
    
    if prompt_type == "default":
        if language == "chinese":
            return """..."""  # ✅ 中文prompt
        elif language == "english":
            return "Please generate a concise English emotion description..."  # ✅ 英文prompt
    
    elif prompt_type == "detailed":
        if language == "chinese":
            return """任务：请生成"情感说明长句"..."""  # ✅ 中文详细prompt
        elif language == "english":
            return "Please provide a detailed analysis..."  # ✅ 英文详细prompt
```

**状态**: ✅ **已支持英语** (通过`language`参数切换)

---

## 📊 问题汇总

### ❌ 需要修改的部分

| 位置 | 问题 | 影响 | 优先级 |
|------|------|------|--------|
| **1. Spec Decoding - CJK检测** | 英语不应用某些约束 | Repetition/n-gram控制 | 🔴 高 |
| **2. Spec Decoding - 标点列表** | 缺少英文标点 | 标点闸门失效 | 🔴 高 |
| **3. Spec Decoding - 标点闸门** | 只统计CJK字符 | 英语标点控制失效 | 🔴 高 |
| **4. BLEU计算** | 硬编码`tokenize='zh'` | 英语BLEU不准确 | 🟡 中 |
| **5. BLEU结果key** | 硬编码`corpus_bleu_zh` | 命名不统一 | 🟢 低 |

### ✅ 已支持英语的部分

| 位置 | 状态 | 说明 |
|------|------|------|
| **Sentence endings** | ✅ | 包含`.` |
| **Stopping criteria** | ✅ | 支持`("。", ".")` |
| **Prompt templates** | ✅ | 通过`language`参数切换 |

---

## 💡 修改建议

### 方案A: 语言感知（Language-Aware）【推荐】

在所有需要的地方添加语言检测，根据语言应用不同逻辑。

#### 优点：
- ✅ 最精确控制
- ✅ 中英文都能获得最优约束
- ✅ 未来易于扩展其他语言

#### 缺点：
- ⚠️ 需要传递`language`参数到生成函数
- ⚠️ 代码复杂度稍高

---

### 方案B: 语言无关（Language-Agnostic）

移除所有CJK特定逻辑，使用通用约束。

#### 优点：
- ✅ 代码简单
- ✅ 自动支持所有语言

#### 缺点：
- ❌ 可能无法解决中文特有问题（标点泛滥等）
- ❌ 性能可能下降

---

### 方案C: 混合方案【推荐实施】

**核心思想**：保留对质量影响大的语言特定逻辑，其他部分使用通用逻辑。

#### 需要修改的优先级：

##### 🔴 **Priority 1: BLEU计算**（必须改）

**位置**: 所有baseline脚本的corpus_bleu计算

**修改**:
```python
# 修改前
corpus_bleu = sacrebleu.corpus_bleu(hyps, refs, tokenize='zh')  ❌

# 修改后
bleu_tokenize = 'zh' if language == 'chinese' else '13a'
corpus_bleu = sacrebleu.corpus_bleu(hyps, refs, tokenize=bleu_tokenize)  ✅

# 结果key也改为动态
bleu_key = f"corpus_bleu_{language[:2]}"  # "corpus_bleu_zh" or "corpus_bleu_en"
```

**原因**: 英语使用中文tokenization会导致BLEU分数不准确。

---

##### 🟡 **Priority 2: Speculative Decoding标点列表**（建议改）

**位置**: `src/speculative_decoding.py` 第979, 1009-1010行

**修改**:
```python
# 修改前（只有中文）
PUNCT_IDS = _ids_for(['，', '。', '、', '：', ':', '；', '！', '？'])
COMMA_LIKE = _ids_for(['，', '、', '：', ':'])
PERIOD_LIKE = _ids_for(['。'])

# 修改后（中英文都有）
PUNCT_IDS = _ids_for([
    '，', '。', '、', '：', '；', '！', '？',  # 中文
    ',', '.', ';', ':', '!', '?'              # 英文
])
COMMA_LIKE = _ids_for([
    '，', '、', '：',  # 中文
    ',', ';', ':'      # 英文
])
PERIOD_LIKE = _ids_for([
    '。',  # 中文
    '.'    # 英文
])
```

**原因**: 英语生成时需要标点闸门防止标点泛滥。

---

##### 🟡 **Priority 3: 标点闸门的字符统计**（建议改）

**位置**: `src/speculative_decoding.py` 第1012-1023行

**修改**:
```python
# 修改前（只统计CJK）
since_punct = 0
for t in reversed(hist):
    if t in PUNCT_IDS:
        break
    s = tokenizer.decode([t])
    if any('\u4e00' <= ch <= '\u9fff' for ch in s):  # ❌ 只统计CJK
        since_punct += 1

# 修改后（统计所有非标点token）
since_punct = 0
for t in reversed(hist):
    if t in PUNCT_IDS:
        break
    # 统计所有非标点token（不限语言）
    since_punct += 1  # ✅ 通用
```

**同时调整阈值**:
```python
# 中文：4字=4tokens, 5字=5tokens
# 英文：4-5词 ≈ 4-5tokens（英语tokenization通常1词=1-2tokens）

# 逗号：保持4 tokens（英语约2-4个词）
if since_punct < 4:
    logits[COMMA_LIKE] = -inf

# 句号：保持5 tokens（英语约3-5个词）
if since_punct < 5:
    logits[PERIOD_LIKE] -= 3.5
```

**原因**: 阈值设计较合理，英语也适用。

---

##### 🟢 **Priority 4: CJK特定约束**（可选改）

**位置**: `src/speculative_decoding.py` 第926-1004行

**选项1: 保持不变**（推荐）
- Repetition penalty、same-char blocking、trigram ban只对CJK生效
- 英语不应用这些约束（英语tokenization不同，不需要）
- **优点**: 简单，且英语baseline本来就没这些约束

**选项2: 改为语言无关**
- 所有语言都应用这些约束
- 需要调整参数（如repetition_penalty强度）
- **缺点**: 可能影响中文质量

**推荐**: 保持不变，英语不需要这些CJK特定的约束。

---

## 🎯 最小修改方案（推荐实施）

只修改**Priority 1和2**，保持其他不变：

### 修改1: BLEU计算

**文件**: 
- `experiments/runs/run_edge_baseline_cpu_limited.py`
- `experiments/runs/run_cloud_baseline.py`
- `experiments/runs/run_speculative_decoding_cpu_limited.py`

**修改内容**:
```python
# 在corpus_bleu计算处
bleu_tokenize = 'zh' if language == 'chinese' else '13a'
corpus_bleu = sacrebleu.corpus_bleu(hyps, refs, tokenize=bleu_tokenize)

# 结果key改为动态
bleu_key = f"corpus_bleu_{language[:2]}"
metrics[bleu_key] = overall_bleu

# 日志也改为动态
logger.info(f"Corpus BLEU ({language} tokenization): {overall_bleu:.4f}")
```

---

### 修改2: Speculative Decoding标点列表

**文件**: `src/speculative_decoding.py`

**修改内容**:
```python
# 第979行
PUNCT_IDS = _ids_for([
    '，', '。', '、', '：', '；', '！', '？',  # Chinese
    ',', '.', ';', ':', '!', '?'              # English
])

# 第1009-1010行
COMMA_LIKE = _ids_for(['，', '、', '：', ',', ';', ':'])
PERIOD_LIKE = _ids_for(['。', '.'])
```

---

### 修改3: 标点闸门统计（可选）

**文件**: `src/speculative_decoding.py`

**修改内容**:
```python
# 第1012-1023行：简化为统计所有非标点token
since_punct = 0
for t in reversed(hist):
    if t in PUNCT_IDS:
        break
    since_punct += 1  # 不限语言
```

---

## 🧪 测试建议

### 测试1: 英语Edge Baseline

```bash
python experiments/runs/run_edge_baseline_cpu_limited.py \
    --dataset_type unified \
    --dataset_path data/processed/secap/manifest.json \
    --caption_type original \
    --language english \
    --prompt_type detailed \
    --max_samples 10 \
    --max_cpu_cores 2 \
    --max_memory_gb 16.0 \
    --output_name edge_cpu_limited_secap_en
```

### 测试2: 英语Cloud Baseline

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

### 验证点

- [ ] BLEU使用英语tokenization (`tokenize='13a'`)
- [ ] 结果key为 `corpus_bleu_en`
- [ ] 生成的英语文本流畅，无标点泛滥
- [ ] Stopping criteria正确（检测到`.`后停止）
- [ ] 输出长度合理（2-3句话）

---

## 📝 总结

### 核心问题

代码中有**3个主要的中文特定部分**：
1. ✅ **Prompt**: 已支持（通过`language`参数）
2. ❌ **BLEU计算**: 硬编码中文tokenization
3. ❌ **标点列表**: 只有中文标点

### 推荐方案

**最小修改**（修改1+2）：
- 修改BLEU tokenization根据language动态选择
- 扩展标点列表包含英文标点
- 保持CJK特定约束不变（英语不受影响）

**预期效果**：
- ✅ 中文baseline继续正常工作
- ✅ 英语baseline能够正确运行
- ✅ 英语BLEU分数准确
- ✅ 英语生成质量合理

**工作量**: 修改3个文件，约20行代码
