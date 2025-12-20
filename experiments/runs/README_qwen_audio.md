# Qwen-Audio Baseline Experiment

This script runs the official MER2025 Qwen-Audio implementation for emotion caption generation.

## Model Difference

- **Qwen-Audio-Chat**: The official model for emotion recognition tasks (used in MER2025)
- **Qwen2-Audio-7B**: General audio processing model (different from Chat version)

We use **Qwen-Audio-Chat** to match the official MER2025 implementation.

## Usage

### Test with 3 samples
```bash
cd /data/gpfs/projects/punim2341/jiajunlu/edgecloud-sec

python experiments/runs/run_qwen_audio_baseline.py \
    --manifest_path data/processed/mer2024/manifest_test.json \
    --model_path Qwen/Qwen-Audio-Chat \
    --output_name qwen_audio_test \
    --max_samples 3 \
    --language english \
    --subtitle_flag subtitle \
    --verbose
```

### Test with 10 samples (English)
```bash
python experiments/runs/run_qwen_audio_baseline.py \
    --manifest_path data/processed/mer2024/manifest_audio_text_augmented_v5.json \
    --model_path Qwen/Qwen-Audio-Chat \
    --output_name qwen_audio_mer_en_test10 \
    --max_samples 10 \
    --language english \
    --subtitle_flag subtitle \
    --verbose
```

### Test with 10 samples (Chinese)
```bash
python experiments/runs/run_qwen_audio_baseline.py \
    --manifest_path data/processed/mer2024/manifest_audio_text_augmented_v5.json \
    --model_path Qwen/Qwen-Audio-Chat \
    --output_name qwen_audio_mer_zh_test10 \
    --max_samples 10 \
    --language chinese \
    --subtitle_flag subtitle \
    --verbose
```

### Full run (English)
```bash
python experiments/runs/run_qwen_audio_baseline.py \
    --manifest_path data/processed/mer2024/manifest_audio_text_augmented_v5.json \
    --model_path Qwen/Qwen-Audio-Chat \
    --output_name qwen_audio_mer_en_full \
    --language english \
    --subtitle_flag subtitle
```

### Full run (Chinese)
```bash
python experiments/runs/run_qwen_audio_baseline.py \
    --manifest_path data/processed/mer2024/manifest_audio_text_augmented_v5.json \
    --model_path Qwen/Qwen-Audio-Chat \
    --output_name qwen_audio_mer_zh_full \
    --language chinese \
    --subtitle_flag subtitle
```

### Without subtitle (audio only)
```bash
python experiments/runs/run_qwen_audio_baseline.py \
    --manifest_path data/processed/mer2024/manifest_audio_text_augmented_v5.json \
    --model_path Qwen/Qwen-Audio-Chat \
    --output_name qwen_audio_mer_en_nosubtitle \
    --language english \
    --subtitle_flag nosubtitle
```

## Parameters

- `--manifest_path`: Path to manifest file (required)
- `--model_path`: Model path (default: `Qwen/Qwen-Audio-Chat`)
- `--output_name`: Output filename (default: `qwen_audio_results`)
- `--max_samples`: Maximum number of samples to process (optional)
- `--language`: Language for generation: `english` or `chinese` (default: `english`)
- `--subtitle_flag`: Use subtitle: `subtitle` or `nosubtitle` (default: `subtitle`)
- `--caption_type`: Caption type: `audio_only` or `original` (default: `audio_only`)
- `--cache_dir`: HuggingFace cache directory
- `--verbose`: Print detailed results

## Output

Results are saved to `experiments/results/{output_name}.json` with the following structure:

```json
{
  "experiment_config": {
    "dataset_type": "unified",
    "caption_type": "audio_only",
    "language": "english",
    "model": "Qwen-Audio"
  },
  "metrics": {
    "corpus_bleu_en": 0.xxxx,
    "avg_bleu_1_sentence": 0.xxxx,
    "avg_bleu_4_sentence": 0.xxxx,
    "avg_meteor_sentence": 0.xxxx,
    "avg_rouge_l_sentence": 0.xxxx,
    "avg_cider": 0.xxxx,
    "avg_bertscore_f1": 0.xxxx,
    "latency_metrics": {
      "ttft_mean": 1.xxx,
      "ttft_min": 0.xxx,
      "ttft_max": 2.xxx,
      ...
    }
  },
  "detailed_results": [...]
}
```

## Official Implementation

The script follows the official MER2025 Qwen-Audio implementation:
- Uses the same prompt as `MERTools/MER2025/MER2025_Track23/Qwen-Audio/main-audio.py`
- Supports both `subtitle` and `nosubtitle` modes
- Uses `Qwen-Audio-Chat` model (not Qwen2-Audio-7B)

## Evaluation Metrics

The script calculates:
- **BLEU** (1-gram and 4-gram, corpus-level and sentence-level)
- **METEOR**
- **ROUGE-L**
- **CIDEr**
- **BERTScore** (precision, recall, F1)

And resource usage:
- **Latency**: TTFT, ITPS, OTPS, OET, Total Time
- **Resources**: CPU%, RAM, GPU utilization, GPU memory

