# Edge vs Cloud Baseline（Qwen-2.5-Omni 3B vs 7B）代码与结果复核（v2）

_Updated: 2025-10-02 16:03 (local)_

本文基于你**修改后的代码与新一轮结果**（`edge_cpu_limited_test15.json`、`cloud_secap_chinese_test20.json`）进行复核：先给出总体结论，然后从代码正确性、结果合理性、不合理点定位与修复建议四个部分展开。文末附上参考资料（sacreBLEU 中文分词、BERTScore 中文/基线重标定、Qwen2.5-Omni 多模态使用、Hugging Face 生成参数、CIDEr 论文）。

---

## TL;DR（总体结论）

- 这轮你**已经修正了关键的评测配置**：云端/边端统一了**中文模板**，并在云端**新增了按中文分词的语料级 BLEU**（`corpus_bleu_zh`）。这是对的。
- 但**流式生成路径**仍未落实**最小生成长度与解码一致化**（如 `min_new_tokens`、`do_sample=False`、`no_repeat_ngram_size`、`repetition_penalty`），导致云端（7B）仍有较多**超短输出**（“一词打卡”式）。
- **BERTScore** 仍有**负值样本**，说明**中文语言/基线重标定**尚未完全配置到位。修正后 7B 的优势会更稳定地体现。

---

## 一、你当前结果的关键现象（来自 JSON 的可验证数据）

**云端（7B） vs 边端（3B）每 100 条样本汇总：**

- 平均输出 token：edge = **8.55**，cloud = **7.96**
- 过短输出（`<= 6` 个汉字）：edge = **8** / 100，cloud = **32** / 100
- 一词输出（`output_tokens <= 1`）：edge = **0**，cloud = **18**
- BERTScore 出现负值的样本数：edge = **12**，cloud = **20**

**整体时延（平均）**：

- TTFT：edge = **9.38s**，cloud = **0.10s**
- OTPS：edge = **0.07 tok/s**，cloud = **3.95 tok/s**
- Total time：edge = **12.77s**，cloud = **0.34s**

**评测分数（聚合）**：

- `corpus_bleu_zh`（语料级、中文分词）：edge = **0.009577**，cloud = **0.009703**
- `avg_cider`（句子级平均）：edge = **0.196470**，cloud = **0.190800**
- `avg_bertscore_f1`（句子级平均）：edge = **0.158105**，cloud = **0.165693**

> 解读：云端 7B 仍存在**较多超短与一词输出**，这会在 BLEU/CIDEr 上被系统性惩罚；BERTScore 也会受文本极短与 baseline 配置影响。

---

## 二、代码检查：哪些已修正？哪些仍需修

### ✅ 已修正/改对
1. **中文模板**统一到“15–40 个汉字”的风格；这有利于更稳定的句式与覆盖信息量。
2. **云端**新增 `corpus_bleu_zh`（sacreBLEU 的中文分词、语料级计算），是正确方向。

### ❗仍需修复
1. **流式生成路径缺少约束**  
   - 云端：`StreamingGenerator.generate_with_accurate_metrics(...)` 只传了 `max_new_tokens, temperature, top_p`，**没有传入** `min_new_tokens`、`do_sample=False`、`no_repeat_ngram_size`、`repetition_penalty` 等限制；fallback 的非流式路径也仍是 `do_sample=True`。  
   - 边端：逐 token 的流式生成采用 `do_sample=True` 且**遇到 EOS 直接停止**，没有“**未达最小新词数就忽略 EOS**”的逻辑。  
   - 这两点直接导致**早停与一词输出**。

2. **BERTScore 配置仍不稳**  
   - 结果中存在**负值样本**，通常是**语言未设为中文（`lang="zh"`）或未启用 `rescale_with_baseline=True`** 导致。建议在批量评测函数中**强制中文 & baseline 重标定**。

3. **（可选）CPU 内存限制**  
   - 监控数据表明“2 核 / 16GB”的名义限制下，实际进程内存均值仍约 21GB。若需严格复现实验环境，建议改用 **cgroup/容器 `--memory`** 或在代码层面减少 KV 缓存与中间张量峰值。

---

## 三、结果是否合理？
- 从**时延**看，云端 7B 具有明显吞吐优势；从**输出长度分布**看，云端仍偏短，**一词输出**比例高达 **18%**。这与**未在流式路径生效的长度/贪心约束**高度一致。
- 在当前设置下，`corpus_bleu_zh` 两边接近并不代表 3B≈7B，而是 7B **被“短答惩罚”低估**。修正生成与评测后，7B 的优势应更清晰。

---

## 四、最小可行修复（按优先级）

### P0：统一/强化生成约束（**流式也要生效**）
- **云端流式**：在 `generate_with_accurate_metrics`（及其内部）增加并传入：
  ```python
  min_new_tokens=8,
  do_sample=False,
  no_repeat_ngram_size=2,
  repetition_penalty=1.05
  ```
  且当遇到 EOS 时，若 `generated_len < min_new_tokens`，**忽略该次 EOS** 继续生成。

- **边端流式**：逐 token 生成改为 **贪心**（`do_sample=False, temperature=0.0, top_p=1.0`），并同样加入**最小新词数**与**忽略早停**逻辑；块生成与一次性生成路径也对齐上述约束。

### P0：BERTScore（中文 + 重标定）
- 在批量评测调用处/内部**固定**：`lang="zh"`（或 `model_type="bert-base-chinese"`），并启用 `rescale_with_baseline=True`。这会显著减少异常值，提升对短文本的鲁棒性。

### P1：评测呈现（更稳健）
- 主报告：**语料级 BLEU（中文分词）**、BERTScore（中文+baseline）、**chrF**（字符级，更不受长度影响）。
- 辅助展示：长度分布（中位数、p90）、句子级 BLEU/CIDEr 仅作诊断，不再直接平均当作总分。

### P2：CPU 限制与内存（可选）
- 若需严格资源约束，使用容器参数/cgroup。Python 层尽量避免保留大中间张量。

---

## 参考资料（便于核查/复现）

- **sacreBLEU 中文分词（TokenizerZh）**：BLEU 语料级、中文需指定 `tokenize='zh'`  
  - https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/tokenizers/tokenizer_zh.py
- **BERTScore（中文/重标定）**：`lang="zh"` 与 `rescale_with_baseline` 的说明与基线表  
  - https://github.com/Tiiiger/bert_score/blob/master/journal/rescale_baseline.md
- **Qwen2.5-Omni（Transformers 文档）**：`apply_chat_template`、多模态输入（含音频）示例  
  - https://huggingface.co/docs/transformers/main/en/model_doc/qwen2_5_omni
- **Hugging Face 生成参数**：`min_length / min_new_tokens`、`do_sample`、`no_repeat_ngram_size`、`repetition_penalty` 等的语义与用法  
  - https://huggingface.co/docs/transformers/v4.18.0/en/main_classes/text_generation
- **CIDEr 原论文（CVPR 2015）**：多参考设计背景，短文本/单参考时需谨慎解读  
  - https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vedantam_CIDEr_Consensus-Based_Image_2015_CVPR_paper.pdf

---

如需，我可以把以上修复打成最小化 **diff 补丁**（覆盖 cloud/edge 的流式与非流式路径、BERTScore 评测函数、runner 的模板统一与长度兜底），并附上 A/B 验证脚本。

