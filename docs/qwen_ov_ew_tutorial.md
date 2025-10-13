
# Qwen‑2.5‑Omni‑7B 抽取 + EW 评测（MER‑OV）一站式教程

> 目标：把**你已生成的英文情感描述**（audio‑only，由 Qwen 产生）转换为**开放词汇情感标签集合**，写成 `predict-openset.csv`，并用 **MER 官方 EW（Emotion‑Wheel）口径**离线评测 **set‑level accuracy / recall**。  
> 本教程默认你只做**英文分支**（最终评测口径要求英文标签）。

---

## 参考与对齐依据（强相关）

- **OV‑MER 论文**：提出 OV‑MER 与两套评测（GPT‑based 与 EW‑based）；附录给出 **#5 OV label extraction** 提示词，用于“从描述抽取开放情感标签”。（Lian et al., 2024）  
- **MER 官方网站**：明确 **MER‑OV 最终标签需为英文**；`final-openset-*.csv` 为从描述抽取得到的标签集合；`check-openset.csv` 为核对后的真值集合。  
- **MERTools 评测仓库**：`MER2024/main-ov.py` 提供本地评测入口；`ov_store/` 下包含 `check-openset.csv`、`openset-synonym.zip`（离线同义/词形归并资产）。

> 提示：官方线上最终排名的 GPT‑based 评测会在服务器上**重复 5 次**取均值；本教程走**EW‑based 离线评测**，完全可复现且零 API 成本。

---

## 一、准备环境与代码（我已经克隆好了，只需要检查即可）

```bash
# 1) 克隆评测代码
git clone https://github.com/zeroQiaoba/MERTools.git
cd MERTools/MER2024

# 2) 确认评测资产
ls ov_store
# 你应能看到：check-openset.csv、predict-openset.csv（模板）、openset-synonym.zip 等
```

> `check-openset.csv`：官方核对后的**英文**真值集合。  
> `openset-synonym.zip`：官方预生成的“同义/词形归并缓存”。使用它即可**离线**跑 EW 评测。

---

## 二、准备你的输入（英文描述）

把你已有的英文情感描述整理成 `qwen_descriptions.csv`：

```csv
name,description
sample_00000648,"The speaker sounds tense and irritable; clipped phrases with abrupt bursts..."
sample_00000680,"Low energy, quivering voice, signs of disappointment and reluctance..."
```

- `name` **必须**与 `check-openset.csv` 的样本名一致（如 `sample_000xxxxx`），否则无法对齐评测。  
- `description` 是你用 Qwen 在上一步生成的**英文情感描述**。

---

## 三、用 Qwen‑2.5‑Omni‑7B 从“描述→标签集合”（#5 提示词）

> 思路：把我从音频中模型输出得到的情感描述（.json文件中的"generated_text"字段）喂给 Qwen‑2.5‑Omni‑7B，让它**只输出英文情感词/短语的 JSON 列表**（1–8 个，去重，小写），这就是每个样本的开放词汇集合。



### 提示词（按论文 #5 语义改写，专注“仅输出标签列表”）

> Please assume the role of an expert in the field of emotions. We provide clues that may be related to the emotions of the characters. Based on the provided clues, please identify the emotional states of the main characters. Please separate different emotional categories with commas and output only the clearly identifiable emotional categories in a list format. If none are identified, please output an empty list.

> 

### 然后输出的标签集合需要保存为 csv，输出文件名字由输入参数定义，输出应该有两列

name, openset，参考仓库中的predict-openset.csv

---

## 四、EW 评测（离线、零成本）

1) 先在 `ov_store/` **解压**官方给的 `openset-synonym.zip`（得到目录 `ov_store/openset-synonym/`，里面是一堆 `*.npy` 的“同义簇”缓存）。  
2) 运行“只计算指标”的入口：

```bash
cd MERTools/MER2024
python main-ov.py calculate_openset_overlap_rate_mer2024 \
  --gt_csv='ov_store/check-openset.csv' \
  --pred_csv='ov_store/predict-openset.csv' \
  --synonym_root='ov_store/openset-synonym'
```

脚本会：  
- 读取 `openset-synonym/*.npy`，把同义/词形映射到同一“组”；  
- 将 GT/Pred 映射到**组集合**后，逐样本计算  
  - `set-accuracy = |GT ∩ Pred| / |Pred|`（Pred 为空记 0）  
  - `set-recall    = |GT ∩ Pred| / |GT|`  
- 输出全体样本的平均 `accuracy`、`recall` 以及它们的均值（常作为汇总分）。

> 如果你**不解压**而直接跑 `main-ov.py main_metric ...`，脚本会尝试**在线**调用 GPT 生成同义簇（会产生成本，不建议）。

---

## 五、常见问题（FAQ）

- **必须用 GPT 抽取吗？** 不需要。论文里 #5 用 GPT‑3.5 是**基线**；你完全可以用 Qwen 抽取，再用 EW 评测。  
- **为什么要英文？** MER‑OV 官方公告与评测真值以**英文**为准；中文分支用于对照，最终对齐仍以英文集合评测。  
- **`predict-openset.csv` 列名？** 对于 MERTools 评测脚本，要用 `name,openset` 两列，`openset` 为**列表字面量字符串**（如 `['anger','disappointment']`）。  
- **数值可比性？** 与官方“GPT‑based 最终分”并非一一等价（那是服务器端 GPT‑3.5 多次均值）。但**EW‑based** 是论文提供的**可复现替代**，适合本地横向对比与 ablation。

---

## 附：最小可运行指令清单（复盘）

```bash
# A. 准备
git clone https://github.com/zeroQiaoba/MERTools.git
cd MERTools/MER2024
# (将 openset-synonym.zip 解压到 ov_store/openset-synonym/)

# B. 你的描述 → 标签集合（用 Qwen 调 API）
cd ../..
python tools/extract_qwen_ov.py \
  qwen_descriptions.json \
  MERTools/MER2024/ov_store/predict-openset.csv

# C. EW 评测（离线）
cd MERTools/MER2024
python main-ov.py calculate_openset_overlap_rate_mer2024 \
  --gt_csv='ov_store/check-openset.csv' \
  --pred_csv='ov_store/predict-openset.csv' \
  --synonym_root='ov_store/openset-synonym'
```

---

### 备注：信息来源与核对点
- OV‑MER 论文（提出 OV‑MER、#5 抽取、两种评测思路；英文标签与集合指标定义）；
- MER 官方网站（英文标签要求；`check-openset.csv` 与 `final-openset-*.csv` 说明；线上评测多次均值）；
- MERTools 仓库（`main-ov.py`、`ov_store/` 结构与离线资产）；
- 通义千问（阿里云）**OpenAI 兼容**接口文档（如何设置 `BASE_URL` 与模型名）；
- vLLM 本地 OpenAI 兼容部署文章（如需自部署）。
```

