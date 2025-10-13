# 英语支持实施计划 - 完整分析与修改方案

## 📋 分析结果总结

### ✅ 已经支持英语的部分

| 组件 | 位置 | 状态 | 说明 |
|------|------|------|------|
| **Prompt模板** | 所有baseline脚本 | ✅ | 通过`--language english`切换 |
| **BERTScore** | `src/evaluation/metrics.py` | ✅ | 中文用`bert-base-chinese`<br>英文用`roberta-large` |
| **Stopping criteria** | `src/models/stopping_criteria.py` | ✅ | 支持`("。", ".")` |
| **Sentence endings** | `src/speculative_decoding.py` | ✅ | 检测`.`和`。` |

### ❌ 需要修改的部分

| 组件 | 位置 | 问题 | 优先级 |
|------|------|------|--------|
| **BLEU tokenization** | 3个baseline脚本 | 硬编码`tokenize='zh'` | 🔴 必须 |
| **标点符号列表** | `speculative_decoding.py` | 只有中文标点 | 🟡 建议 |
| **标点闸门统计** | `speculative_decoding.py` | 只统计CJK字符 | 🟡 建议 |
| **CJK特定约束** | `speculative_decoding.py` | Repetition/n-gram只对CJK | 🟢 保持不变 |

---

## 🔧 详细修改计划

### 修改1: BLEU Tokenization（🔴 必须修改）

#### 影响文件（3个）
1. `experiments/runs/run_edge_baseline_cpu_limited.py`
2. `experiments/runs/run_cloud_baseline.py`
3. `experiments/runs/run_speculative_decoding_cpu_limited.py`

#### 当前代码

**Edge Baseline** (第535行):
```python
corpus_bleu = sacrebleu.corpus_bleu(hyps, refs, tokenize='zh')  ❌
overall_bleu = corpus_bleu.score / 100.0

"corpus_bleu_zh": overall_bleu  ❌
logger.info(f"Corpus BLEU (Chinese tokenization): {overall_bleu:.4f}")  ❌
```

**Cloud Baseline** (第265行): 相同问题

**Spec Decoding** (第573行): 相同问题

#### 修改后代码

```python
# 动态选择tokenization
bleu_tokenize = 'zh' if language == 'chinese' else '13a'
corpus_bleu = sacrebleu.corpus_bleu(hyps, refs, tokenize=bleu_tokenize)  ✅
overall_bleu = corpus_bleu.score / 100.0

# 动态key名称
bleu_key = f"corpus_bleu_{language[:2]}"  # "zh" or "en"
{bleu_key: overall_bleu}  ✅

# 动态日志
lang_display = "Chinese" if language == 'chinese' else "English"
logger.info(f"Corpus BLEU ({lang_display} tokenization): {overall_bleu:.4f}")  ✅
```

**BERTScore说明**（见README）:
- 中文：`tokenize='zh'` - 使用中文分词
- 英文：`tokenize='13a'` - 使用标准英文tokenization（处理标点、大小写等）

---

### 修改2: 标点符号列表（🟡 建议修改）

#### 影响文件
- `src/speculative_decoding.py`

#### 当前代码（第979, 1009-1010行）

```python
PUNCT_IDS = _ids_for(['，', '。', '、', '：', ':', '；', '！', '？'])  ❌ 只有中文
COMMA_LIKE = _ids_for(['，', '、', '：', ':'])  ❌
PERIOD_LIKE = _ids_for(['。'])  ❌
```

**问题**：英语生成时，标点闸门不起作用（因为英文标点不在列表中）

#### 修改后代码

```python
# 包含中英文标点
PUNCT_IDS = _ids_for([
    # Chinese punctuation
    '，', '。', '、', '：', '；', '！', '？',
    # English punctuation
    ',', '.', ';', ':', '!', '?'
])

COMMA_LIKE = _ids_for([
    # Chinese
    '，', '、', '：',
    # English
    ',', ';', ':'
])

PERIOD_LIKE = _ids_for([
    # Chinese
    '。',
    # English
    '.'
])
```

**效果**：
- ✅ 中文和英文都能应用标点闸门
- ✅ 防止英语也出现逗号泛滥

---

### 修改3: 标点闸门字符统计（🟡 建议修改）

#### 影响文件
- `src/speculative_decoding.py`

#### 当前代码（第1012-1023行）

```python
# 只统计CJK字符
since_punct = 0
for t in reversed(hist):
    if t in PUNCT_IDS:
        break
    s = tokenizer.decode([t])
    if any('\u4e00' <= ch <= '\u9fff' for ch in s):  # ❌ 只统计CJK
        since_punct += 1
```

**问题**：英语token不会被统计，since_punct始终为0，标点闸门不起作用

#### 修改后代码

```python
# 统计所有非标点token（不限语言）
since_punct = 0
for t in reversed(hist):
    if t in PUNCT_IDS:
        break
    # 统计所有token（中英文通用）
    since_punct += 1  ✅
```

**阈值分析**：
- 中文：4个token = 4个汉字 = 合理
- 英文：4个token ≈ 2-4个单词 = 合理（英语tokenization通常1词=1-2 tokens）
- **结论**：阈值无需调整，对中英文都适用

---

### 修改4: CJK特定约束（🟢 保持不变）

#### 代码位置
- Repetition penalty (第926-945行)
- Same-char blocking (第981-986行)
- Content-only trigram (第992-1004行)

#### 决定：**不修改**

**原因**：
1. 这些约束是为了解决**中文特有问题**（标点泛滥、单字重复）
2. 英语tokenization不同（通常1词=1 token），不会有这些问题
3. 英语会自动跳过这些约束（`_is_cjk()`返回False）

**验证**：
- 英语token的`_is_cjk()`检测会返回False
- Repetition penalty不会应用到英语
- Same-char blocking不会应用到英语
- Trigram ban不会应用到英语
- **这是正确的行为！**

---

## 📊 修改总结表

| 修改项 | 文件 | 行号 | 优先级 | 工作量 |
|--------|------|------|--------|--------|
| **BLEU tokenize** | `run_edge_baseline_cpu_limited.py` | 535 | 🔴 必须 | 5行 |
| **BLEU tokenize** | `run_cloud_baseline.py` | 265 | 🔴 必须 | 5行 |
| **BLEU tokenize** | `run_speculative_decoding_cpu_limited.py` | 573 | 🔴 必须 | 5行 |
| **标点列表** | `speculative_decoding.py` | 979, 1009-1010 | 🟡 建议 | 10行 |
| **闸门统计** | `speculative_decoding.py` | 1012-1023 | 🟡 建议 | 5行 |

**总工作量**: 约30行代码修改

---

## 🎯 实施顺序

### Phase 1: BLEU修复（必须，立即实施）

修改3个baseline脚本的BLEU计算，使其根据language参数动态选择tokenization。

**预期效果**：
- ✅ 英语BLEU分数准确
- ✅ 中文BLEU不受影响

---

### Phase 2: 标点控制（建议，可选）

修改Speculative Decoding的标点列表和统计逻辑。

**预期效果**：
- ✅ 英语也能应用标点闸门
- ✅ 防止英语标点泛滥

---

### Phase 3: 测试验证

运行英语baseline验证修改效果。

---

## 🧪 测试数据集

根据您打开的文件，您有SECAP数据集：

**数据集**: `data/processed/secap/manifest.json`

**测试命令**：

```bash
# Edge Baseline - English
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

# Cloud Baseline - English  
python experiments/runs/run_cloud_baseline.py \
    --dataset_type unified \
    --dataset_path data/processed/secap/manifest.json \
    --caption_type original \
    --language english \
    --prompt_type detailed \
    --max_samples 10 \
    --output_name cloud_secap_en
```

---

## 📝 验证清单

修改后，检查以下各项：

### BLEU验证
- [ ] 结果中有 `corpus_bleu_en` key（不是`corpus_bleu_zh`）
- [ ] BLEU分数合理（英语通常0.15-0.30）
- [ ] 日志显示"English tokenization"

### BERTScore验证
- [ ] 使用`roberta-large`模型（不是`bert-base-chinese`）
- [ ] 分数合理（英语通常0.85-0.95）

### 生成质量验证
- [ ] 英语句子流畅，语法正确
- [ ] 无标点泛滥（", , , ,"）
- [ ] 2-3句话（符合detailed prompt）
- [ ] 无对话式内容

### 中文baseline验证
- [ ] 中文baseline继续正常工作
- [ ] BLEU分数与之前一致
- [ ] 无regression

---

## 💡 关键洞察

### BERTScore的语言支持（根据README）

| 语言 | Model Type | 说明 |
|------|-----------|------|
| **英语 (en)** | `roberta-large` | 默认英语模型 |
| **中文 (zh)** | `bert-base-chinese` | 默认中文模型 |
| **其他** | `bert-base-multilingual-cased` | 多语言模型 |

**我们的实现**：
- ✅ 中文：`hfl/chinese-roberta-wwm-ext-large`（更好的中文模型）
- ✅ 英文：`roberta-large`（标准）
- ✅ 通过`language`参数自动选择

---

### BLEU的Tokenization（根据sacreBLEU文档）

| 语言 | Tokenization | 说明 |
|------|--------------|------|
| **中文** | `'zh'` | 字符级分词 |
| **英语** | `'13a'` | 标准英语（处理标点、大小写） |
| **国际** | `'intl'` | 国际化tokenization |

**我们需要修改**：
- ❌ 当前：所有语言都用`'zh'`
- ✅ 修改后：根据`language`参数动态选择

---

## 🎯 推荐实施方案

### 最小修改方案（推荐）

**只修改BLEU计算**，其他保持不变：

1. ✅ **修改BLEU tokenization**（3个文件）
   - 根据language参数动态选择
   - 中文：`'zh'`
   - 英文：`'13a'`

2. 🟢 **保持标点控制不变**
   - 标点列表暂时不改（先测试）
   - 如果英语出现标点问题，再扩展列表

**原因**：
- BERTScore已经支持英语 ✅
- Prompt已经支持英语 ✅
- 只有BLEU是硬编码的 ❌
- 标点控制可以在测试后按需添加

---

### 完整修改方案（可选）

如果测试发现英语有标点问题，再添加：

1. ✅ 扩展标点列表（包含英文标点）
2. ✅ 改进标点闸门统计（统计所有token）

---

## 🚀 下一步

**需要我帮您实施修改吗？**

我将：
1. ✅ 修改3个baseline脚本的BLEU计算
2. ✅ （可选）扩展Speculative Decoding的标点列表
3. ✅ 验证代码无语法错误
4. ✅ 提供测试命令

**预计修改时间**: 10分钟
**预计修改行数**: 15-30行

---

## 📝 文档

已创建：
- `docs/chinese_specific_code_analysis.md` - 中文特定代码分析
- `docs/english_support_implementation_plan.md` - 本文档

**准备好开始修改了吗？** 🚀

