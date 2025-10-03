# Edge vs Cloud Baseline: 为什么 7B（云）没有明显优于 3B（边）？
_Last updated: 2025-10-02 13:43 _

本文把我对你当前实验（**edge: Qwen-2.5-Omni-3B / cloud: Qwen-2.5-Omni-7B**）的分析整理成可复用的文档，包含：结论、证据（来自你的结果 JSON）、修改建议与补丁位点。

---

## TL;DR（结论）

- **评测方法学的偏置** 和 **生成长度差异** 是 7B 没拉开差距的主因：
  - 你目前在**逐样本**上计算 **BLEU** 并取平均，且**中文没有使用对应的分词器**；这对**“更短、更克制”**的云端 7B 输出形成系统性惩罚。
  - **CIDEr** 在**单参考**时波动很大，同样偏向较长的句子。
  - **BERTScore** 配置很可能没有按**中文**模式/模型来算，导致出现**负值**等异常。

- 模型管线（Omni 的 chat+audio 模式）在云和边**总体一致**；关键差别是**采样 + 长度设置**与**评测配置**，而不是“有没有把音频送进模型”。

---

## 证据（来自你的结果文件）

> 数据来源：`edge_cpu_limited_test14.json` 与 `cloud_secap_chinese_test19.json`。

### 1) 聚合指标（100 样本）

- **BLEU（句子级平均）**：edge = **0.0214**，cloud = **0.0141**  
- **CIDEr（句子级平均）**：edge = **0.1988**，cloud = **0.1937**  
- **BERTScore F1（句子级平均）**：edge = **0.1586**，cloud = **0.1664**（此处云端略优）

> 注：BLEU/CIDEr 的“逐样本再平均”会强化**短答惩罚**；BERTScore 的**语言模型/基线**若不匹配会产生异常值，影响平均。

### 2) 长度与延迟（你的 latency_metrics）

- **平均输出 token 数**：edge = **9.21**，cloud = **5.77**（云端更短）  
- **平均输入 token 数**：两端一致 = **415.79**  
- **TTFT**：edge = **9.48s** vs cloud = **0.075s**  
- **OTPS**：edge = **0.072 tok/s** vs cloud = **4.03 tok/s**  
- **总时长（Total time）**：edge = **13.11s** vs cloud = **0.268s**

> 说明：云端 7B 的输出更**短**（平均 5.77 vs 9.21），这对 BLEU/CIDEr 的句子级取平均是**吃亏**的。

### 3) 生成内容特征

- 云端有 **44 / 100** 条生成只有 6 个字符以内（如“愤怒”“高兴”等**单词/短词**）。  
- **BERTScore recall 为负**的条目：cloud = **21**，edge = **8**（强信号：**语言/基线配置不当**）。

---

## 诊断与解释

1. **BLEU 的用法不对等**  
   - BLEU 设计为**语料级**指标；逐句 BLEU 后再平均会**高方差**、**短答更吃亏**。  
   - 中文必须使用适当的**tokenize（如 `zh`）**；否则中文会被当作“整句一个 token”或错误切分，分数失真。

2. **CIDEr 的场景不匹配**  
   - CIDEr 是为**多参考的图像描述**而设计；在**单参考**和**短句**时波动更大，且偏向更长的生成。

3. **BERTScore 的中文配置**  
   - 如果没有指定 `lang="zh"` 或中文模型（如 `bert-base-chinese`）以及**baseline 重标定**，就会出现**负值**或整体偏差。

4. **生成策略导致 7B 更短**  
   - 两端都在使用 `do_sample=True, temperature=0.7, top_p=0.9`，但 7B 更“克制”，经常给出**单词级**输出，进而在 n-gram 指标上被系统性扣分。

---

## 修改建议（一步到位的 checklist）

### A. 让 BLEU/CIDEr 更公平
- **BLEU：改为“语料级 + 中文分词”**  
  使用 sacreBLEU：`sacrebleu.corpus_bleu(hyps, [refs], tokenize='zh')`。  
  **不要**再把逐句 BLEU 平均当作最终分数；逐句 BLEU 可留作**诊断日志**。

- **CIDEr：单参考谨慎使用**  
  如无法提供多参考，保留 CIDEr 但**降权**，并补充 **chrF** 或 **BERTScore** 这类更鲁棒的指标。

### B. 把 BERTScore 配到“中文正确姿势”
- 显式指定：`bert_score.score(cands, refs, lang="zh", rescale_with_baseline=True)`，或 `model_type="bert-base-chinese"`。  
- 继续开启 IDF（你已经在批量评测里使用 IDF 了，这点很好）。

### C. 统一/约束生成长度
- 解码参数两端对齐，并加**长度下限**：
  ```python
  do_sample=False,        # 或更低温度
  min_new_tokens=8,
  max_new_tokens=32,
  no_repeat_ngram_size=2,
  repetition_penalty=1.05
  ```
- 轻微改 Prompt，加入**长度锚点**（两端一致）：  
  “请用**8–16 个字**的中文短句描述此段音频的情绪（只写情绪及语气特征，不要具体事件）。”

### D. 报告方式
- 增加**语料级 BLEU**、**chrF**、**BERTScore（中文配置）**的**总体分**，并报告**长度分布**（中位数/分位数），减少极端样本影响。
- 同时输出**平均/中位输出长度**，帮助判断模型是否“过度简短”。

---

## 补丁位点（你可以直接改这些文件）

1. **BLEU 改为语料级 + 中文分词**  
   - 文件：`run_cloud_baseline.py` 与 `run_edge_baseline_cpu_limited.py`  
   - 把：
     ```python
     avg_bleu = sum(r['bleu_score'] for r in results) / len(results)
     ```
     换成：
     ```python
     import sacrebleu
     hyps = [r['generated_text'] for r in results]
     refs = [[r['reference_caption'] for r in results]]
     corpus_bleu = sacrebleu.corpus_bleu(hyps, refs, tokenize='zh')
     overall_bleu = corpus_bleu.score / 100.0  # 与[0,1]区间对齐
     ```

2. **BERTScore（中文）**
   - 在评测模块里把 `language="chinese"` **映射为** `lang="zh"`；并开启：`rescale_with_baseline=True`；或直接：`model_type="bert-base-chinese"`。

3. **统一生成策略（边/云）**
   - 文件：`cloud_model.py::generate_independently` 与 `edge_model.py::generate_draft`  
   - 生成时加入：
     ```python
     do_sample=False,
     min_new_tokens=8,
     max_new_tokens=32,
     no_repeat_ngram_size=2,
     repetition_penalty=1.05,
     ```
   - Prompt 加入**8–16 个字**长度提示。

---

## 预期变化

- **7B（云）**会在 **BERTScore / chrF** 等鲁棒指标上显著领先，且**语料级 BLEU（中文分词）**也不再被短答系统性压低。  
- 两端因**解码/长度**不同带来的比较偏差将被最大程度消除。

---

## 附：为什么这些修改是“标准做法”？（资料）

- **Qwen2.5-Omni** 官方文档：多模态（含**音频**）输入、对话模板与推理范式（Transformers & 官方仓库）。  
  - HF Transformers 文档（Qwen2.5-Omni）  
    <https://huggingface.co/docs/transformers/main/en/model_doc/qwen2_5_omni>  
  - 官方仓库（QwenLM/Qwen2.5-Omni）  
    <https://github.com/QwenLM/Qwen2.5-Omni>

- **BLEU（sacreBLEU）**：语料级计算；中文需要 `tokenize='zh'`（或等效中文分词）。  
  - sacreBLEU 的中文分词器实现（TokenizerZh）  
    <https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/tokenizers/tokenizer_zh.py>

- **BERTScore（中文）**：使用 `lang="zh"` 或中文模型，且建议开启 baseline 重标定。  
  - README（rescale baseline 原理与配置）  
    <https://github.com/Tiiiger/bert_score/blob/master/journal/rescale_baseline.md>  
  - 中文 baseline 文件（示例）  
    <https://github.com/Tiiiger/bert_score/blob/master/bert_score/rescale_baseline/zh/bert-base-chinese.tsv>

- **CIDEr**：为**多参考图像描述**设计；在**单参考/短句**时波动较大，应谨慎解读。  
  - 原论文（CVPR 2015）  
    <https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vedantam_CIDEr_Consensus-Based_Image_2015_CVPR_paper.pdf>

---

如需，我可以把以上改动做成最小可复用的 **patch（diff）**，直接贴到对应文件中。

