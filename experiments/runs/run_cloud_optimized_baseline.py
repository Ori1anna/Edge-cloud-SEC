#!/usr/bin/env python3
"""
Cloud Optimized Baseline Experiment Script

This script runs the Cloud Optimized Baseline experiment using the SAME generation logic
as Edge Baseline and Speculative Decoding. This ensures fair comparison between:
1. Edge Optimized Baseline (3B model with spec logic)
2. Cloud Optimized Baseline (7B model with spec logic) 
3. Speculative Decoding (Edge + Cloud verification)

The Cloud Optimized Baseline uses GPU for inference and the same optimized generation
logic as Speculative Decoding, enabling us to truly evaluate the performance difference
between 7B Cloud model vs 3B Edge model.
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
import numpy as np
from datasets import load_dataset
import pandas as pd
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from models.cloud_model import CloudModel
from evaluation.metrics import EvaluationMetrics
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_test_data(data_path: str, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load test dataset
    
    Args:
        data_path: Path to the test dataset
        num_samples: Number of samples to load (None for all)
        
    Returns:
        List of test samples
    """
    logger.info(f"Loading test data from: {data_path}")
    
    if data_path.endswith('.json'):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # Assume it's a dataset name
        dataset = load_dataset(data_path)
        data = dataset['test'] if 'test' in dataset else dataset['validation']
        data = data.to_list()
    
    if num_samples:
        data = data[:num_samples]
        logger.info(f"Limited to {num_samples} samples")
    
    logger.info(f"Loaded {len(data)} test samples")
    return data

def get_caption_field(sample: Dict[str, Any], caption_type: str, language: str) -> str:
    """Get the appropriate caption field based on type and language"""
    if caption_type == "original":
        return sample.get("caption", "")
    elif caption_type == "audio_only":
        if language == "chinese":
            return sample.get("chinese_caption", "")
        elif language == "english":
            return sample.get("english_caption", "")
        else:
            raise ValueError(f"Unsupported language: {language}")
    else:
        raise ValueError(f"Unsupported caption type: {caption_type}")

def get_prompt_template(prompt_type: str = "default", language: str = "chinese") -> str:
    """
    Get prompt template based on type and language (same as original cloud baseline)
    
    Args:
        prompt_type: Type of prompt ("default", "detailed", "concise")
        language: Language for generation ("chinese", "english")
        
    Returns:
        Prompt template string
    """
    if prompt_type == "default":
        if language == "chinese":
            return """任务：请基于给定音频，输出一句"情感说明短句"。

必须遵守：
- 只输出一句中文短句（12–30个汉字），以"。"结尾。
- 句子中同时包含：一个主要情绪 + 一个简短的声学/韵律线索（如语气、语速、强弱、音高变化等"类别"层面的描述即可），但不要解释或列举。
- 不要出现客套话、邀请继续对话、表情符号、英文、Markdown、标号或代码；不要提及"音频/模型/分析/我"。
- 若存在多种可能性，只选择最可能的一种，不要并列罗列。

只给出最终这"一句短句"，不要输出其他内容。"""
        elif language == "english":
            return "Please generate a concise English emotion description based on the audio content. Example: 'The speaker's voice trembles, expressing sadness and disappointment.'"
    elif prompt_type == "detailed":
        if language == "chinese":
            return """任务：请生成"情感说明长句"，按以下顺序组织内容并保持自然流畅：

(1) 先用2–3个"类别级"的声学/韵律线索描述说话方式（从以下维度中任选若干：语速、音调高低/起伏、音量强弱、停顿与连贯度、音色/紧张度等），不用给数值；
(2) 据此给出最可能的单一情绪（不列举选项）；
(3) 若语义内容暗示缘由，可用极简的一小短语点到为止（可用"可能/似乎/大概"表不确定）。

输出要求：
- 只输出"两到三句中文长句"，约70–100个字；
- 使用第三人称或"说话人"等指代；不要出现第一/第二人称；不要设问或邀请对话；
- 不要编造具体人物/时间/地点等细节；不要出现表情符号、英文、Markdown/代码。"""
        elif language == "english":
            return "As an expert in the field of emotions, please focus on the acoustic information in the audio to discern clues related to the emotions of the individual. Please provide a detailed description and ultimately predict the emotional state of the individual. Please provide your answer in English only."
    elif prompt_type == "concise":
        if language == "chinese":
            return "请用最简洁的中文描述音频中的情感状态。"
        elif language == "english":
            return "Please describe the emotional state in the audio using the most concise English."
    else:
        raise ValueError(f"Unsupported prompt type: {prompt_type}")

def run_cloud_optimized_baseline_experiment(
    config_path: str = "configs/default.yaml",
    dataset_type: str = "unified",
    dataset_path: str = None,
    output_name: str = "cloud_optimized_results",
    max_samples: int = None,
    verbose: bool = False,
    caption_type: str = "original",
    language: str = "chinese",
    prompt_type: str = "default",
    input_modality: str = "audio_only",
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Run Cloud Optimized Baseline experiment using speculative decoding logic
    
    Args:
        config_path: Path to configuration file
        dataset_type: Type of dataset ("unified")
        dataset_path: Path to dataset manifest file
        output_name: Name for output files
        max_samples: Maximum number of samples to process
        verbose: Whether to print detailed results
        caption_type: Type of caption to use ("original", "audio_only")
        language: Language for generation ("chinese", "english")
        prompt_type: Type of prompt to use ("default", "detailed", "concise")
        input_modality: Input modality ("audio_only", "audio_text")
        device: Device to use for inference
        
    Returns:
        Dictionary containing experiment results
    """
    logger.info("=" * 80)
    logger.info("CLOUD OPTIMIZED BASELINE EXPERIMENT")
    logger.info("Using Speculative Decoding logic for fair comparison")
    logger.info("=" * 80)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load dataset
    dataset = load_test_data(dataset_path, max_samples)
    
    # Limit samples if specified
    if max_samples:
        dataset = dataset[:max_samples]
        logger.info(f"Limited to {max_samples} samples")
    
    logger.info(f"Caption type: {caption_type}")
    logger.info(f"Language: {language}")
    logger.info(f"Prompt type: {prompt_type}")
    logger.info(f"Input modality: {input_modality}")
    logger.info(f"Using device: {device}")
    
    # Initialize Cloud model
    cloud_model = CloudModel(
        model_name="Qwen/Qwen2.5-Omni-7B",  # Use same model as original cloud baseline
        device=device,
        dtype="float32"  # Updated to float32 for fair comparison
    )
    
    logger.info("Cloud model initialized successfully")
    
    # Initialize metrics calculator
    metrics = EvaluationMetrics()
    
    # Get prompt template
    prompt_template = get_prompt_template(prompt_type, language)
    logger.info(f"Using prompt template: {prompt_template[:100]}...")
    
    # Initialize results storage
    results = []
    all_metrics = []
    
    # Processing statistics
    total_samples = len(dataset)
    processed_samples = 0
    failed_samples = 0
    
    logger.info(f"Starting experiment with {total_samples} samples")
    
    # Process each sample
    for i, sample in enumerate(dataset):
        try:
            logger.info(f"Processing sample {i+1}/{total_samples}: {sample.get('file_id', f'sample_{i}')}")
            
            # Load audio data
            audio_path = sample['audio_path']
            if not os.path.exists(audio_path):
                logger.error(f"Audio file not found: {audio_path}")
                failed_samples += 1
                continue
            
            # Load audio waveform
            import librosa
            audio_waveform, _ = librosa.load(audio_path, sr=16000)
            audio_waveform = torch.tensor(audio_waveform, dtype=torch.float32)
            
            # Generate text using Cloud Optimized Baseline (same logic as Edge baseline)
            logger.info(f"Generating text for sample {sample.get('file_id', f'sample_{i}')}")
            start_time = time.time()
            
            # Use same parameters as Edge baseline for fair comparison
            generated_text, latency_metrics = cloud_model.generate_with_spec_logic(
                audio_features=audio_waveform,
                prompt=prompt_template,
                max_new_tokens=128,         # Same as Edge baseline
                target_sentences=2,         # Same as Edge baseline
                min_chars=90,               # Same as Edge baseline
                min_new_tokens_sc=48,       # Same as Edge baseline
                prompt_type=prompt_type     # Pass prompt type for stopping criteria
            )
            
            generation_time = time.time() - start_time
            
            # Clean generated text: remove newlines, tabs, and strip
            # This is critical for pycocoevalcap METEOR which writes to temp files
            generated_text = generated_text.replace('\n', ' ').replace('\t', ' ').strip()
            
            # Get reference text based on caption type
            reference_text = get_caption_field(sample, caption_type, language)
            
            # Calculate evaluation metrics (same as Edge baseline)
            logger.info(f"Calculating metrics for sample {sample.get('file_id', f'sample_{i}')}")
            bleu_1_score = metrics.compute_bleu_1([reference_text], generated_text, language=language)
            bleu_4_score = metrics.compute_bleu_4([reference_text], generated_text, language=language)
            meteor_score = metrics.compute_meteor([reference_text], generated_text, language=language)
            rouge_l_score = metrics.compute_rouge_l([reference_text], generated_text)
            cider_score = metrics.compute_cider([reference_text], generated_text, language=language)
            
            # Store metrics in a dictionary format
            sample_metrics = {
                'bleu_1': bleu_1_score,
                'bleu_4': bleu_4_score,
                'meteor': meteor_score,
                'rouge_l': rouge_l_score,
                'cider': cider_score
            }
            
            # Store results
            result = {
                'file_id': sample.get('file_id', f'sample_{i}'),
                'audio_path': audio_path,
                'reference_text': reference_text,
                'generated_text': generated_text,
                'generation_time': generation_time,
                'latency_metrics': latency_metrics,
                'evaluation_metrics': sample_metrics
            }
            
            results.append(result)
            all_metrics.append(sample_metrics)
            
            processed_samples += 1
            
            # Log progress
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i+1}/{total_samples} samples ({processed_samples} successful, {failed_samples} failed)")
            
            # Log sample result
            logger.info(f"Sample {sample.get('file_id', f'sample_{i}')} completed:")
            logger.info(f"  - Generated text: {generated_text[:100]}...")
            logger.info(f"  - Generation time: {generation_time:.3f}s")
            logger.info(f"  - TTFT: {latency_metrics.get('ttft', 0.0):.3f}s")
            logger.info(f"  - OTPS: {latency_metrics.get('otps', 0.0):.3f} tokens/s")
            logger.info(f"  - BLEU-4: {sample_metrics.get('bleu_4', 0.0):.4f}")
            logger.info(f"  - METEOR: {sample_metrics.get('meteor', 0.0):.4f}")
            
        except Exception as e:
            logger.error(f"Error processing sample {sample.get('file_id', f'sample_{i}')}: {e}")
            import traceback
            traceback.print_exc()
            failed_samples += 1
            continue
    
    # Calculate aggregate metrics (same as Edge baseline)
    logger.info("Calculating aggregate metrics...")
    
    if all_metrics:
        # Calculate corpus-level metrics using sacrebleu (same as other baselines)
        import sacrebleu
        hyps = [r['generated_text'] for r in results]
        refs_list = [r['reference_text'] for r in results]
        refs = [[r['reference_text'] for r in results]]
        
        # Select tokenization based on language
        bleu_tokenize = 'zh' if language == 'chinese' else '13a'
        corpus_bleu_result = sacrebleu.corpus_bleu(hyps, refs, tokenize=bleu_tokenize)
        
        # Extract BLEU-4 and BLEU-1
        corpus_bleu_4 = corpus_bleu_result.score / 100.0
        corpus_bleu_1 = corpus_bleu_result.precisions[0] / 100.0
        
        # Corpus-level METEOR, ROUGE-L, CIDEr
        corpus_meteor = metrics.compute_corpus_meteor(hyps, refs_list, language=language)
        corpus_rouge_l = metrics.compute_corpus_rouge_l(hyps, refs_list)
        corpus_cider = metrics.compute_corpus_cider(hyps, refs_list, language=language)
        
        # Calculate sentence-level metrics (averages)
        avg_bleu_1 = np.mean([m.get('bleu_1', 0.0) for m in all_metrics])
        avg_bleu_4 = np.mean([m.get('bleu_4', 0.0) for m in all_metrics])
        avg_meteor = np.mean([m.get('meteor', 0.0) for m in all_metrics])
        avg_rouge_l = np.mean([m.get('rouge_l', 0.0) for m in all_metrics])
        avg_cider = np.mean([m.get('cider', 0.0) for m in all_metrics])
        
        # Calculate latency metrics
        avg_ttft = np.mean([r['latency_metrics'].get('ttft', 0.0) for r in results])
        avg_otps = np.mean([r['latency_metrics'].get('otps', 0.0) for r in results])
        avg_total_time = np.mean([r['generation_time'] for r in results])
        avg_output_tokens = np.mean([r['latency_metrics'].get('output_tokens', 0) for r in results])
        
        # Extract detailed latency metrics (same as cloud baseline)
        detailed_latency_data = [r['latency_metrics'] for r in results if r['latency_metrics']]
        latency_metrics = metrics.compute_detailed_latency_metrics(detailed_latency_data)
        
        aggregate_metrics = {
            'average_bleu_1': avg_bleu_1,
            'average_bleu_4': avg_bleu_4,
            'average_meteor': avg_meteor,
            'average_rouge_l': avg_rouge_l,
            'average_cider': avg_cider,
            'average_ttft': avg_ttft,
            'average_otps': avg_otps,
            'average_total_time': avg_total_time,
            'average_output_tokens': avg_output_tokens,
            'total_samples': total_samples,
            'processed_samples': processed_samples,
            'failed_samples': failed_samples,
            'success_rate': processed_samples / total_samples if total_samples > 0 else 0.0
        }
    else:
        logger.warning("No metrics calculated - all samples failed")
        latency_metrics = {}
        aggregate_metrics = {
            'total_samples': total_samples,
            'processed_samples': 0,
            'failed_samples': failed_samples,
            'success_rate': 0.0
        }
        # Set corpus-level metrics to 0.0 for consistency
        corpus_bleu_1 = 0.0
        corpus_bleu_4 = 0.0
        corpus_meteor = 0.0
        corpus_rouge_l = 0.0
        corpus_cider = 0.0
    
    # Prepare final results (same structure as cloud baseline)
    final_results = {
        'experiment_config': {
            'dataset_type': dataset_type,
            'dataset_path': dataset_path,
            'caption_type': caption_type,
            'language': language,
            'prompt_type': prompt_type,
            'max_samples': max_samples,
            'total_samples': len(results)
        },
        'metrics': {
            # Corpus-level metrics
            f'corpus_bleu_1_{language[:2]}': corpus_bleu_1,
            f'corpus_bleu_4_{language[:2]}': corpus_bleu_4,
            'corpus_meteor': corpus_meteor,
            'corpus_rouge_l': corpus_rouge_l,
            'corpus_cider': corpus_cider,
            # Sentence-level metrics (averaged)
            'avg_bleu_1_sentence': aggregate_metrics.get('average_bleu_1', 0.0),
            'avg_bleu_4_sentence': aggregate_metrics.get('average_bleu_4', 0.0),
            'avg_meteor_sentence': aggregate_metrics.get('average_meteor', 0.0),
            'avg_rouge_l_sentence': aggregate_metrics.get('average_rouge_l', 0.0),
            'avg_cider': aggregate_metrics.get('average_cider', 0.0),
            # Latency metrics
            'latency_metrics': latency_metrics
        },
        'detailed_results': results
    }
    
    # Save results
    os.makedirs("experiments/results", exist_ok=True)
    output_file = os.path.join("experiments/results", f"{output_name}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to: {output_file}")
    lang_display = "Chinese" if language == 'chinese' else "English"
    
    # Log corpus-level metrics
    logger.info("=" * 80)
    logger.info("CORPUS-LEVEL METRICS:")
    logger.info("=" * 80)
    logger.info(f"Corpus BLEU-1 ({lang_display} tokenization): {final_results['metrics'][f'corpus_bleu_1_{language[:2]}']:.4f}")
    logger.info(f"Corpus BLEU-4 ({lang_display} tokenization): {final_results['metrics'][f'corpus_bleu_4_{language[:2]}']:.4f}")
    logger.info(f"Corpus METEOR: {final_results['metrics']['corpus_meteor']:.4f}")
    logger.info(f"Corpus ROUGE-L: {final_results['metrics']['corpus_rouge_l']:.4f}")
    logger.info(f"Corpus CIDEr: {final_results['metrics']['corpus_cider']:.4f}")
    
    # Log sentence-level metrics
    logger.info("=" * 80)
    logger.info("SENTENCE-LEVEL METRICS (AVERAGE):")
    logger.info("=" * 80)
    logger.info(f"Average BLEU-1: {final_results['metrics']['avg_bleu_1_sentence']:.4f}")
    logger.info(f"Average BLEU-4: {final_results['metrics']['avg_bleu_4_sentence']:.4f}")
    logger.info(f"Average METEOR: {final_results['metrics']['avg_meteor_sentence']:.4f}")
    logger.info(f"Average ROUGE-L: {final_results['metrics']['avg_rouge_l_sentence']:.4f}")
    logger.info(f"Average CIDEr: {final_results['metrics']['avg_cider']:.4f}")
    
    # Log simplified latency metrics
    if latency_metrics:
        logger.info("Latency Metrics (Mean/Min/Max):")
        logger.info(f"  TTFT: {latency_metrics.get('ttft_mean', 0):.4f}s (min: {latency_metrics.get('ttft_min', 0):.4f}s, max: {latency_metrics.get('ttft_max', 0):.4f}s)")
        logger.info(f"  ITPS: {latency_metrics.get('itps_mean', 0):.2f} tokens/sec (min: {latency_metrics.get('itps_min', 0):.2f}, max: {latency_metrics.get('itps_max', 0):.2f})")
        logger.info(f"  OTPS: {latency_metrics.get('otps_mean', 0):.2f} tokens/sec (min: {latency_metrics.get('otps_min', 0):.2f}, max: {latency_metrics.get('otps_max', 0):.2f})")
        logger.info(f"  OET: {latency_metrics.get('oet_mean', 0):.4f}s (min: {latency_metrics.get('oet_min', 0):.4f}s, max: {latency_metrics.get('oet_max', 0):.4f}s)")
        logger.info(f"  Total Time: {latency_metrics.get('total_time_mean', 0):.4f}s (min: {latency_metrics.get('total_time_min', 0):.4f}s, max: {latency_metrics.get('total_time_max', 0):.4f}s)")
        logger.info("Resource Usage (Mean/Min/Max):")
        logger.info(f"  CPU: {latency_metrics.get('cpu_percent_mean', 0):.1f}% (min: {latency_metrics.get('cpu_percent_min', 0):.1f}%, max: {latency_metrics.get('cpu_percent_max', 0):.1f}%)")
        logger.info(f"  RAM: {latency_metrics.get('ram_gb_mean', 0):.2f}GB (min: {latency_metrics.get('ram_gb_min', 0):.2f}GB, max: {latency_metrics.get('ram_gb_max', 0):.2f}GB)")
        logger.info(f"  GPU: {latency_metrics.get('gpu_util_mean', 0):.1f}% (min: {latency_metrics.get('gpu_util_min', 0):.1f}%, max: {latency_metrics.get('gpu_util_max', 0):.1f}%)")
        logger.info(f"  GPU Memory: {latency_metrics.get('gpu_memory_gb_mean', 0):.2f}GB (min: {latency_metrics.get('gpu_memory_gb_min', 0):.2f}GB, max: {latency_metrics.get('gpu_memory_gb_max', 0):.2f}GB)")
    
    return final_results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run Cloud Optimized Baseline experiment")
    parser.add_argument("--config", default="configs/default.yaml", help="Configuration file path")
    parser.add_argument("--dataset_type", default="unified", help="Dataset type")
    parser.add_argument("--dataset_path", required=True, help="Path to dataset manifest file")
    parser.add_argument("--output_name", default="cloud_optimized_results", help="Output file name")
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples to process")
    parser.add_argument("--verbose", action="store_true", help="Print detailed results")
    parser.add_argument("--caption_type", default="original", choices=["original", "audio_only"], 
                       help="Type of caption to use")
    parser.add_argument("--language", default="chinese", choices=["chinese", "english"], 
                       help="Language for generation")
    parser.add_argument("--prompt_type", default="default", choices=["default", "detailed", "concise"], 
                       help="Type of prompt to use")
    parser.add_argument("--input_modality", default="audio_only", choices=["audio_only", "audio_text"],
                       help="Input modality: 'audio_only' (audio only) or 'audio_text' (audio + transcription)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for inference")
    
    args = parser.parse_args()
    
    # Run experiment
    results = run_cloud_optimized_baseline_experiment(
        config_path=args.config,
        dataset_type=args.dataset_type,
        dataset_path=args.dataset_path,
        output_name=args.output_name,
        max_samples=args.max_samples,
        verbose=args.verbose,
        caption_type=args.caption_type,
        language=args.language,
        prompt_type=args.prompt_type,
        input_modality=args.input_modality,
        device=args.device
    )
    
    logger.info("Experiment completed successfully!")

if __name__ == "__main__":
    main()
