#!/usr/bin/env python3
"""
Generic Cloud-Only Baseline Experiment Runner
Supports multiple datasets, prompts, and languages through configuration
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import pandas as pd
import torch
import yaml

from src.data.audio_processor import AudioProcessor
from src.models.cloud_model import CloudModel
from src.evaluation.metrics import EvaluationMetrics

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_unified_manifest(manifest_path: str) -> List[Dict[str, Any]]:
    """Load unified manifest format"""
    logger.info(f"Loading unified manifest from {manifest_path}")
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    logger.info(f"Loaded {len(manifest)} samples from unified manifest")
    return manifest

def load_dataset(dataset_type: str, dataset_path: str) -> List[Dict[str, Any]]:
    """Load dataset based on type"""
    if dataset_type == "unified":
        return load_unified_manifest(dataset_path)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

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

def get_prompt_template(prompt_type: str, language: str) -> str:
    """Get prompt template based on type and language"""
    if prompt_type == "default":
        if language == "chinese":
            return """你是“音频情感描述器”。只做一件事：依据音频的可听线索，输出一条简洁的中文情感描述。

【输出格式（必须严格遵守）】
- 只输出一行中文短句；不加引号；不用任何前缀（如“示例/编号/输出/结果”）；不使用英文或“视频/文本”等字样；不提问、不寒暄、不解释过程。
- 句式写法：用“情感词 + 声学线索 + 态度倾向”的描述性短语式表达进行输出。
- 长度建议 15–40 个汉字。

- 严禁输出：“无法判断情绪”“请提供更多信息”“好的/嗯”等对话式内容。
- 如线索不充分，也要**就近选择最可能的情感**并给出至少一条声学线索。
- 不捏造具体人物/事件/语义内容，只依据声音特征与可听到的情绪状态作描述。

【你的唯一任务】
- 从音频中提取可听到的情绪与相关声学线索，按上述格式输出一条中文描述。
- 只输出这“一行描述”，除此之外不要输出任何其他文字。"""


#             return """根据音频，仅输出一条中文情感描述。
# 要求：只一行，句号结尾；不提问、不对话、不解释；不用引号、英文或数字。
# 内容只含“情感词 + 声学线索”（1–3 个情感 + 至少 1 条线索），用逗号或顿号连接。
# 情感词可用：悲伤、愤怒、开心、平静、焦虑、担忧、害怕、失望、嫉妒、冷淡、厌倦、轻蔑、真诚、自信、激动等；
# 声学线索可用：语调上扬、语调下降、语速加快、语速放慢、声音洪亮、声音低沉、声音颤抖、停顿明显、带哭腔、含笑、有气无力、铿锵有力、低声、压抑等。
# 即使线索较弱，也写出最可能的情感并附一条声学线索。"""

    elif language == "english":
            return "Please generate a concise English emotion description based on the audio content. Example: 'The speaker's voice trembles, expressing sadness and disappointment.'"
    elif prompt_type == "detailed":
        if language == "chinese":
            return "请详细分析音频中的情感特征，包括语调、语速、音量等，并生成详细的中文情感描述。"
        elif language == "english":
            return "Please provide a detailed analysis of emotional features in the audio, including tone, speed, volume, etc., and generate a detailed English emotion description."
    elif prompt_type == "concise":
        if language == "chinese":
            return "请用最简洁的中文描述音频中的情感状态。"
        elif language == "english":
            return "Please describe the emotional state in the audio using the most concise English."
    else:
        raise ValueError(f"Unsupported prompt type: {prompt_type}")

def run_cloud_baseline_experiment(config_path: str = "configs/default.yaml", 
                                dataset_type: str = "unified",
                                dataset_path: str = None,
                                output_name: str = "cloud_only_results",
                                max_samples: int = None, 
                                verbose: bool = False,
                                caption_type: str = "original",
                                language: str = "chinese",
                                prompt_type: str = "default"):
    """
    Run cloud-only baseline experiment with flexible configuration
    
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
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load dataset
    dataset = load_dataset(dataset_type, dataset_path)
    
    # Limit samples if specified
    if max_samples:
        dataset = dataset[:max_samples]
        logger.info(f"Processing {max_samples} samples")
    else:
        logger.info(f"Processing {len(dataset)} samples")
    
    logger.info(f"Caption type: {caption_type}")
    logger.info(f"Language: {language}")
    logger.info(f"Prompt type: {prompt_type}")
    
    # Setup device and models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    audio_processor = AudioProcessor(**config['audio'])
    cloud_model = CloudModel(**config['models']['cloud'])
    metrics = EvaluationMetrics()
    
    # Get prompt template
    prompt_template = get_prompt_template(prompt_type, language)
    logger.info(f"Using prompt template: {prompt_template}")
    
    # Process samples
    results = []
    
    for i, sample in enumerate(dataset):
        try:
            logger.info(f"Processing sample {i+1}/{len(dataset)}: {sample['file_id']}")
            
            # Load and process audio
            audio_path = sample['audio_path']
            audio_waveform = audio_processor.load_audio(audio_path)
            
            # Get reference caption
            reference_caption = get_caption_field(sample, caption_type, language)
            if not reference_caption:
                logger.warning(f"No reference caption found for {sample['file_id']}")
                continue
            
            # Generate text with detailed latency metrics
            generated_text, detailed_latency = cloud_model.generate_independently(audio_waveform, prompt_template, max_new_tokens=64)
            
            # Calculate traditional metrics
            bleu_score = metrics.compute_bleu([reference_caption], generated_text)
            cider_score = metrics.compute_cider([reference_caption], generated_text)
            
            # Store results (BERTScore will be calculated in batch later)
            result = {
                "file_id": sample['file_id'],
                "dataset": sample['dataset'],
                "reference_caption": reference_caption,
                "generated_text": generated_text,
                "bleu_score": bleu_score,
                "cider_score": cider_score,
                "caption_type": caption_type,
                "language": language,
                "prompt_type": prompt_type,
                "latency_metrics": detailed_latency
            }
            results.append(result)
            
            if verbose:
                logger.info(f"Sample {i+1}:")
                logger.info(f"  Reference: {reference_caption}")
                logger.info(f"  Generated: {generated_text}")
                logger.info(f"  BLEU: {bleu_score:.4f}")
                logger.info(f"  CIDEr: {cider_score:.4f}")
                logger.info(f"  BERTScore: Computing in batch...")
                if detailed_latency:
                    logger.info(f"  TTFT: {detailed_latency.get('ttft', 0):.4f}s")
                    logger.info(f"  ITPS: {detailed_latency.get('itps', 0):.2f} tokens/sec")
                    logger.info(f"  OTPS: {detailed_latency.get('otps', 0):.2f} tokens/sec")
                    logger.info(f"  OET: {detailed_latency.get('oet', 0):.4f}s")
                    logger.info(f"  Total Time: {detailed_latency.get('total_time', 0):.4f}s")
                    logger.info(f"  CPU: {detailed_latency.get('cpu_percent', 0):.1f}%")
                    logger.info(f"  RAM: {detailed_latency.get('ram_gb', 0):.2f}GB")
                    logger.info(f"  GPU: {detailed_latency.get('gpu_util', 0):.1f}%")
                    logger.info(f"  GPU Memory: {detailed_latency.get('gpu_memory_gb', 0):.2f}GB")
                logger.info("")
            
        except Exception as e:
            logger.error(f"Error processing sample {sample['file_id']}: {e}")
            continue
    
    # Calculate overall metrics
    if results:
        # Compute batch BERTScore with corpus-level IDF
        logger.info("Computing BERTScore with corpus-level IDF...")
        candidates = [r['generated_text'] for r in results]
        references_list = [[r['reference_caption']] for r in results]
        bertscore_results = metrics.compute_batch_bertscore(candidates, references_list, language=language)
        
        # Add BERTScore results to each result and log them
        for i, result in enumerate(results):
            result['bertscore_precision'] = bertscore_results[i]['bertscore_precision']
            result['bertscore_recall'] = bertscore_results[i]['bertscore_recall']
            result['bertscore_f1'] = bertscore_results[i]['bertscore_f1']
            
            # Log BERTScore for each sample
            logger.info(f"Sample {i+1} BERTScore:")
            logger.info(f"  Precision: {result['bertscore_precision']:.4f}")
            logger.info(f"  Recall: {result['bertscore_recall']:.4f}")
            logger.info(f"  F1: {result['bertscore_f1']:.4f}")
        
        # Calculate corpus-level BLEU with Chinese tokenization
        import sacrebleu
        hyps = [r['generated_text'] for r in results]
        refs = [[r['reference_caption'] for r in results]]
        corpus_bleu = sacrebleu.corpus_bleu(hyps, refs, tokenize='zh')
        overall_bleu = corpus_bleu.score / 100.0  # Convert to [0,1] range
        
        # Keep sentence-level averages for diagnostic purposes
        avg_bleu_sentence = sum(r['bleu_score'] for r in results) / len(results)
        avg_cider = sum(r['cider_score'] for r in results) / len(results)
        avg_bertscore_precision = sum(r['bertscore_precision'] for r in results) / len(results)
        avg_bertscore_recall = sum(r['bertscore_recall'] for r in results) / len(results)
        avg_bertscore_f1 = sum(r['bertscore_f1'] for r in results) / len(results)
        
        # Extract detailed latency metrics
        detailed_latency_data = [r['latency_metrics'] for r in results if r['latency_metrics']]
        latency_metrics = metrics.compute_detailed_latency_metrics(detailed_latency_data)
        
        overall_results = {
            "experiment_config": {
                "dataset_type": dataset_type,
                "dataset_path": dataset_path,
                "caption_type": caption_type,
                "language": language,
                "prompt_type": prompt_type,
                "max_samples": max_samples,
                "total_samples": len(results)
            },
            "metrics": {
                "corpus_bleu_zh": overall_bleu,  # Corpus-level BLEU with Chinese tokenization
                "avg_bleu_sentence": avg_bleu_sentence,  # Sentence-level average for diagnostics
                "avg_cider": avg_cider,
                "avg_bertscore_precision": avg_bertscore_precision,
                "avg_bertscore_recall": avg_bertscore_recall,
                "avg_bertscore_f1": avg_bertscore_f1,
                "latency_metrics": latency_metrics
            },
            "detailed_results": results
        }
        
        # Save results
        output_file = f"experiments/results/{output_name}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(overall_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to: {output_file}")
        logger.info(f"Corpus BLEU (Chinese tokenization): {overall_bleu:.4f}")
        logger.info(f"Average BLEU (sentence-level): {avg_bleu_sentence:.4f}")
        logger.info(f"Average CIDEr: {avg_cider:.4f}")
        logger.info(f"Average BERTScore Precision: {avg_bertscore_precision:.4f}")
        logger.info(f"Average BERTScore Recall: {avg_bertscore_recall:.4f}")
        logger.info(f"Average BERTScore F1: {avg_bertscore_f1:.4f}")
        
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
        
        return overall_results
    else:
        logger.error("No results generated")
        return None

def main():
    parser = argparse.ArgumentParser(description="Run cloud-only baseline experiment")
    parser.add_argument("--config", default="configs/default.yaml", help="Configuration file path")
    parser.add_argument("--dataset_type", default="unified", help="Dataset type")
    parser.add_argument("--dataset_path", required=True, help="Path to dataset manifest file")
    parser.add_argument("--output_name", default="cloud_only_results", help="Output file name")
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples to process")
    parser.add_argument("--verbose", action="store_true", help="Print detailed results")
    parser.add_argument("--caption_type", default="original", choices=["original", "audio_only"], 
                       help="Type of caption to use")
    parser.add_argument("--language", default="chinese", choices=["chinese", "english"], 
                       help="Language for generation")
    parser.add_argument("--prompt_type", default="default", choices=["default", "detailed", "concise"], 
                       help="Type of prompt to use")
    
    args = parser.parse_args()
    
    run_cloud_baseline_experiment(
        config_path=args.config,
        dataset_type=args.dataset_type,
        dataset_path=args.dataset_path,
        output_name=args.output_name,
        max_samples=args.max_samples,
        verbose=args.verbose,
        caption_type=args.caption_type,
        language=args.language,
        prompt_type=args.prompt_type
    )

if __name__ == "__main__":
    main()
