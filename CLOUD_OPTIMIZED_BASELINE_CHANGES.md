# Cloud Optimized Baseline - 修改说明

## 概述

根据您的要求，我已经修改了Cloud Optimized Baseline，使其使用与原始Cloud Baseline完全相同的模型参数和prompt设置，确保公平比较。

## 主要修改

### 1. 模型初始化参数

**修改前：**
```python
cloud_model = CloudModel(
    model_path=model_path,
    device=device,
    torch_dtype=torch.float16,
    load_in_8bit=False,
    trust_remote_code=True
)
```

**修改后（与原始cloud baseline一致）：**
```python
cloud_model = CloudModel(
    model_name=model_path,
    device=device,
    dtype="float16"  # Same as original cloud baseline
)
```

### 2. Prompt模板

**修改前：** 使用简化的英文prompt

**修改后（与原始cloud baseline一致）：** 使用完整的中文prompt模板

```python
def get_prompt_template(prompt_type: str = "default", language: str = "chinese") -> str:
    if prompt_type == "default":
        if language == "chinese":
            return """任务：请基于给定音频，输出一句"情感说明短句"。

必须遵守：
- 只输出一句中文短句（12–30个汉字），以"。"结尾。
- 句子中同时包含：一个主要情绪 + 一个简短的声学/韵律线索（如语气、语速、强弱、音高变化等"类别"层面的描述即可），但不要解释或列举。
- 不要出现客套话、邀请继续对话、表情符号、英文、Markdown、标号或代码；不要提及"音频/模型/分析/我"。
- 若存在多种可能性，只选择最可能的一种，不要并列罗列。

只给出最终这"一句短句"，不要输出其他内容。"""
```

### 3. 生成参数

**修改前：**
- `max_new_tokens=128`
- `prompt_type="detailed"`

**修改后（与原始cloud baseline一致）：**
- 根据prompt类型动态调整max_new_tokens：
  - `detailed`: 120 tokens
  - `concise`: 32 tokens  
  - `default`: 64 tokens
- `prompt_type="default"`

### 4. SLURM脚本参数

**修改前：**
```bash
--max_new_tokens 128 \
--prompt_type "detailed" \
```

**修改后（与原始cloud baseline一致）：**
```bash
--max_new_tokens 64 \
--prompt_type "default" \
```

## 文件修改列表

1. **`experiments/runs/run_cloud_optimized_baseline.py`**
   - 更新了模型初始化参数
   - 更新了prompt模板函数
   - 更新了生成参数逻辑

2. **`slurm/run_cloud_optimized_baseline.slurm`**
   - 更新了命令行参数

3. **`test_cloud_optimized_baseline.py`**
   - 更新了测试脚本的模型初始化
   - 更新了测试prompt

## 现在的比较设置

| 方法 | 模型 | 生成逻辑 | Prompt | Max Tokens |
|------|------|----------|--------|------------|
| **Edge Optimized Baseline** | 3B Edge | Speculative Decoding逻辑 | 中文default | 动态调整 |
| **Cloud Optimized Baseline** | 7B Cloud | Speculative Decoding逻辑 | 中文default | 动态调整 |
| **原始Cloud Baseline** | 7B Cloud | 标准HF生成逻辑 | 中文default | 动态调整 |
| **Speculative Decoding** | 3B+7B | Speculative Decoding逻辑 | 中文default | 动态调整 |

## 关键优势

1. **公平比较**：Cloud Optimized Baseline现在使用与原始Cloud Baseline完全相同的模型参数和prompt设置
2. **生成逻辑差异**：唯一的差异是生成逻辑（Speculative Decoding vs 标准HF生成）
3. **有意义的结果**：能够真正评估Speculative Decoding生成逻辑相对于标准生成逻辑的优势

## 使用方法

```bash
# 提交作业
sbatch slurm/run_cloud_optimized_baseline.slurm

# 或直接运行
python experiments/runs/run_cloud_optimized_baseline.py \
    --model_path "/data/gpfs/projects/punim2341/jiajunlu/models/Qwen3-4B-Thinking-2507" \
    --test_data "/data/gpfs/projects/punim2341/jiajunlu/datasets/mer_audio_emotion_chinese/test.json" \
    --output_dir "./experiments/results" \
    --num_samples 332 \
    --prompt_type "default" \
    --device "cuda"
```

现在Cloud Optimized Baseline与原始Cloud Baseline使用完全相同的参数设置，唯一的差异是生成逻辑，这将提供真正有意义的比较结果。
