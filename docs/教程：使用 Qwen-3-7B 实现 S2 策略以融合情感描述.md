# 教程：使用 Qwen-3-7B 实现 S2 策略以融合情感描述



本教程将提供一个分步指南，用于复现研究论文 "Open-vocabulary Multimodal Emotion Recognition" (arXiv:2410.01495v1) 中提出的 S2 策略。您将使用 **Qwen-3-7B-Instruct** 模型，将基于音频的情感描述与相应的字幕文本进行融合，以生成一份全面、综合的分析文本。



## 1. 背景知识：什么是 S2 策略？



在论文的上下文中，S2 是一种两步走的策略，旨在通过降低语言模型任务的复杂性来生成高质量、多模态的情感描述 (`CLUE-MLLM`) 。  



- **第一步：单模态描述提取。** 首先，一个专门的多模态大语言模型（MLLM）处理单一的模态（如音频或视频），提取相关的情感线索并生成一段描述性文本。在您的情况下，您已经使用 **Qwen-2.5-Omni** 处理音频模态，完成了这一步。
- **第二步：融合与综合分析。** 接着，将第一步生成的描述性文本与原始文本（字幕）相结合。使用另一个强大的大语言模型（LLM）来分析和整合这两个文本源，从而产生最终的、更全面的描述。

论文发现，这种“分而治之”的方法（S2）通常优于将所有模态一次性输入给单个模型的策略（S1），因为它简化了推理过程 。本教程将重点实现该策略的**第二步**。  





## 2. 目标



您的目标是利用 Qwen-2.5-Omni 生成的纯音频情感描述，并使用 Qwen-3-7B 将其与原始视频字幕进行合并，为每个视频片段生成最终的综合分析。



## 3. 准备工作

### 输入数据



两个 JSON 文件：

**音频描述文件**：您上一个步骤中使用 Qwen-2.5-Omni 的输出文件。

- 该文件是一个 JSON 数组。
- 每个对象必须包含一个用于视频片段的 `file_id` 和位于 `"generated_text"` 字段下的描述文本。

**带字幕的数据集文件**：主数据集文件，其中包含原始字幕。

- 一个 JSON 数组。
- 每个对象包含 `file_id` 和位于 `"english_transcription"` 字段下的英语字幕文本。



## 4. 核心 Prompt 设计



一个精心设计的 prompt 对于引导 LLM 至关重要。我们将根据原始论文的附录（附录D，表6，Prompt #3）调整 prompt，以适应我们融合*听觉*和*文本*线索的特定任务 。  



**为 S2 融合适配后的 Prompt:**

```
Please act as an expert in the field of emotions. We provide acoustic clues that may be related to the character's emotional state, along with the original subtitle of the video. Please analyze which parts can infer the emotional state and explain the reasons. During the analysis, please integrate the textual and audio clues.

Acoustic Clues:
{audio_description}

Original Subtitle:
{subtitle}

Integrated Analysis:
```

这个 prompt 清晰地定义了模型的角色（“情感领域的专家”），提供了结构化的输入（“听觉线索”、“原始字幕”），并指明了期望的输出（“综合分析”）。



## 5. Python 脚本



以下是执行 S2 融合的完整脚本，可以作为参考，但是需要根据我的实际项目的其他代码做一些微调，比如id要变成file_id等等。所以你在写代码时不要照抄，下面代码保存为 `run_s2_fusion.py`。

Python

```
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import argparse

# --- 1. 配置 ---
MODEL_NAME = "Qwen/Qwen3-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. Prompt 模板 ---
# 根据论文适配 (arXiv:2410.01495v1, Appendix D, Table 6)
PROMPT_TEMPLATE = """
Please act as an expert in the field of emotions. We provide acoustic clues that may be related to the character's emotional state, along with the original subtitle of the video. Please analyze which parts can infer the emotional state and explain the reasons. During the analysis, please integrate the textual and audio clues.

Acoustic Clues:
{audio_description}

Original Subtitle:
{subtitle}

Integrated Analysis:
"""

def load_data(audio_desc_path, subtitle_path):
    """根据 'id' 加载并合并音频描述和字幕数据。"""
    print("正在加载并合并数据...")
    try:
        with open(audio_desc_path, 'r', encoding='utf-8') as f:
            # 音频描述位于 'generated_text' 字段
            audio_data = {item['id']: item['generated_text'] for item in json.load(f)}
        
        with open(subtitle_path, 'r', encoding='utf-8') as f:
            # 字幕位于 'english_transcription' 字段
            subtitle_data = {item['id']: item['english_transcription'] for item in json.load(f)}
    except FileNotFoundError as e:
        print(f"错误：找不到文件 {e.filename}")
        return None
    except KeyError as e:
        print(f"错误：其中一个 JSON 文件中缺少预期的键 {e}。")
        return None
        
    merged_data =
    for video_id, subtitle_text in subtitle_data.items():
        if video_id in audio_data:
            merged_data.append({
                "id": video_id,
                "subtitle": subtitle_text,
                "audio_description": audio_data[video_id]
            })
    print(f"成功合并 {len(merged_data)} 条数据。")
    return merged_data

def generate_descriptions(data, model, tokenizer):
    """使用 Qwen-3-7B 模型生成最终的综合描述。"""
    results =
    for item in tqdm(data, desc="正在生成综合描述"):
        full_prompt = PROMPT_TEMPLATE.format(
            audio_description=item['audio_description'],
            subtitle=item['subtitle']
        )
        
        # 为指令微调模型格式化 prompt
        messages =
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)

        # 使用推荐的参数生成文本以鼓励推理
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.1
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        results.append({
            "id": item['id'],
            "final_description": response.strip()
        })
        
    return results

def main():
    parser = argparse.ArgumentParser(description="使用 Qwen-3-7B 运行 S2 策略融合。")
    parser.add_argument('--audio_desc_file', type=str, required=True, help="包含音频描述的 JSON 文件路径。")
    parser.add_argument('--subtitle_file', type=str, required=True, help="包含字幕的 JSON 文件路径。")
    parser.add_argument('--output_file', type=str, default='s2_final_descriptions.json', help="保存输出的 JSON 文件路径。")
    args = parser.parse_args()

    print(f"正在使用的设备: {DEVICE}")

    # 加载模型和分词器
    print(f"正在加载模型: {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("模型加载成功。")

    # 加载并处理数据
    input_data = load_data(args.audio_desc_file, args.subtitle_file)
    if not input_data:
        return

    # 生成最终描述
    final_results = generate_descriptions(input_data, model, tokenizer)
    
    # 保存结果
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
        
    print(f"\n处理完成！最终描述已保存至 {args.output_file}")

if __name__ == "__main__":
    main()
```



## 6. 如何运行脚本



脚本将首先加载模型（这可能需要一些时间），然后处理每个项目，并显示一个进度条。



## 7. 预期输出



脚本完成后，您将得到一个名为 `s2_final_descriptions.json` 的新文件。它将包含每个音频片段的综合分析。