#!/usr/bin/env python3
"""
Qwen-Audio Baseline Experiment Runner
使用官方MER2025的Qwen-Audio实现生成情感描述并评估
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import psutil
import pynvml

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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

def get_transcription_field(sample: Dict[str, Any], language: str) -> str:
    """Get transcription field based on language"""
    if language == "chinese":
        return sample.get("chinese_transcription", "")
    elif language == "english":
        return sample.get("english_transcription", "")
    else:
        raise ValueError(f"Unsupported language: {language}")

def get_qwen_audio_prompt(subtitle: str, subtitle_flag: str, language: str) -> str:
    """
    Get Qwen-Audio prompt based on subtitle flag and language
    
    Args:
        subtitle: The transcription/subtitle content
        subtitle_flag: "subtitle" or "nosubtitle"
        language: "chinese" or "english"
    """
    if subtitle_flag == "subtitle":
        if language == "chinese":
            prompt = f"音频字幕内容：{subtitle}；作为一名情感分析专家，请专注于音频中的声学信息和字幕内容，识别与个人情感相关的线索。请提供详细的描述，并最终预测音频中个体的情感状态。"
        else:  # english
            prompt = f"Subtitle content of the audio: {subtitle}; As an expert in the field of emotions, please focus on the acoustic information and subtitle content in the audio to discern clues related to the emotions of the individual. Please provide a detailed description and ultimately predict the emotional state of the individual in the audio."
    else:  # nosubtitle
        if language == "chinese":
            prompt = "作为一名情感分析专家，请专注于音频中的声学信息，识别与个人情感相关的线索。请提供详细的描述，并最终预测音频中个体的情感状态。"
        else:  # english
            prompt = "As an expert in the field of emotions, please focus on the acoustic information in the audio to discern clues related to the emotions of the individual. Please provide a detailed description and ultimately predict the emotional state of the individual in the audio."
    return prompt

def run_qwen_audio_experiment(
    manifest_path: str,
    model_path: str,
    output_name: str = "qwen_audio_results",
    max_samples: int = None,
    verbose: bool = False,
    caption_type: str = "audio_only",
    language: str = "english",
    subtitle_flag: str = "subtitle",
    cache_dir: str = "/data/gpfs/projects/punim2341/jiajunlu/hf-cache"
):
    """
    Run Qwen-Audio baseline experiment
    
    Args:
        manifest_path: Path to manifest file
        model_path: Path to Qwen-Audio model
        output_name: Name for output files
        max_samples: Maximum number of samples to process
        verbose: Whether to print detailed results
        caption_type: Type of caption to use ("original", "audio_only")
        language: Language for generation ("chinese", "english")
        subtitle_flag: Whether to use subtitle ("subtitle", "nosubtitle")
        cache_dir: HuggingFace cache directory
    """
    # Load dataset
    dataset = load_unified_manifest(manifest_path)
    
    # Limit samples if specified
    if max_samples:
        dataset = dataset[:max_samples]
        logger.info(f"Processing {max_samples} samples")
    else:
        logger.info(f"Processing {len(dataset)} samples")
    
    logger.info(f"Caption type: {caption_type}")
    logger.info(f"Language: {language}")
    logger.info(f"Subtitle flag: {subtitle_flag}")
    
    # Initialize GPU monitoring
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    
    # Load Qwen-Audio-Chat model (same as official MER2025 implementation)
    logger.info(f"Loading Qwen-Audio-Chat model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="cuda",
        trust_remote_code=True,
        cache_dir=cache_dir
    ).eval()
    logger.info("Model loaded successfully")
    
    # Initialize metrics
    metrics = EvaluationMetrics()
    
    # Process samples
    results = []
    all_metrics = []
    
    for i, sample in enumerate(dataset):
        try:
            logger.info(f"Processing sample {i+1}/{len(dataset)}: {sample['file_id']}")
            
            # Get audio path
            audio_path = sample.get('audio_path', '')
            if not os.path.exists(audio_path):
                logger.warning(f"Audio file not found: {audio_path}")
                continue
            
            # Get reference caption
            reference_caption = get_caption_field(sample, caption_type, language)
            if not reference_caption:
                logger.warning(f"No reference caption found for {sample['file_id']}")
                continue
            
            # Get transcription for subtitle
            subtitle = get_transcription_field(sample, language)
            
            # Monitor resource usage before generation
            process = psutil.Process(os.getpid())
            cpu_before = process.cpu_percent(interval=0.1)
            ram_before = process.memory_info().rss / (1024**3)  # GB
            
            # Get GPU usage before
            gpu_info_before = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_mem_before = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024**3)  # GB
            
            # Record start time
            start_time = time.time()
            first_token_time = None
            
            # Build Qwen-Audio query
            prompt = get_qwen_audio_prompt(subtitle, subtitle_flag, language)
            query = tokenizer.from_list_format([
                {'audio': audio_path},
                {'text': prompt},
            ])
            
            # Generate with Qwen-Audio (same as official)
            response, history = model.chat(tokenizer, query=query, history=None)
            generated_text = response.replace('\n', ' ').replace('\t', ' ').strip()
            
            # Record end time
            total_time = time.time() - start_time
            
            # Monitor resource usage after generation
            cpu_after = process.cpu_percent(interval=0.1)
            ram_after = process.memory_info().rss / (1024**3)  # GB
            
            # Get GPU usage after
            gpu_info_after = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_mem_after = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024**3)  # GB
            
            # Calculate average resource usage
            cpu_percent = (cpu_before + cpu_after) / 2
            ram_gb = (ram_before + ram_after) / 2
            gpu_util = (gpu_info_before.gpu + gpu_info_after.gpu) / 2
            gpu_memory_gb = (gpu_mem_before + gpu_mem_after) / 2
            
            # Token counting
            input_tokens = len(query)
            output_tokens = len(tokenizer.encode(generated_text))
            
            # Calculate latency metrics (simplified for Qwen-Audio)
            # Since we can't get precise first token time, we estimate
            ttft = total_time / 2  # Estimate: assume equal time for processing and generation
            otps = output_tokens / (total_time - ttft) if (total_time - ttft) > 0 else 0
            itps = input_tokens / ttft if ttft > 0 else 0
            oet = total_time - ttft  # Output evaluation time
            
            # Calculate evaluation metrics
            bleu_1 = metrics.compute_bleu_1([reference_caption], generated_text, language=language)
            bleu_4 = metrics.compute_bleu_4([reference_caption], generated_text, language=language)
            meteor = metrics.compute_meteor([reference_caption], generated_text, language=language)
            rouge_l = metrics.compute_rouge_l([reference_caption], generated_text)
            cider = metrics.compute_cider([reference_caption], generated_text, language=language)
            
            # BERTScore
            bertscore_result = metrics.compute_bertscore(
                [reference_caption], generated_text, language=language
            )
            
            # Store results
            sample_result = {
                'file_id': sample['file_id'],
                'dataset': sample.get('dataset', 'unknown'),
                'reference_caption': reference_caption,
                'generated_text': generated_text,
                'bleu_1_score': bleu_1,
                'bleu_4_score': bleu_4,
                'meteor_score': meteor,
                'rouge_l_score': rouge_l,
                'cider_score': cider,
                'caption_type': caption_type,
                'language': language,
                'prompt_type': subtitle_flag,
                'latency_metrics': {
                    'ttft': ttft,
                    'itps': itps,
                    'otps': otps,
                    'oet': oet,
                    'total_time': total_time,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'cpu_percent': cpu_percent,
                    'ram_gb': ram_gb,
                    'gpu_util': gpu_util,
                    'gpu_memory_gb': gpu_memory_gb
                },
                'bertscore_precision': bertscore_result.get('bertscore_precision', 0.0),
                'bertscore_recall': bertscore_result.get('bertscore_recall', 0.0),
                'bertscore_f1': bertscore_result.get('bertscore_f1', 0.0)
            }
            
            results.append(sample_result)
            all_metrics.append(sample_result)
            
            # Log progress
            if verbose:
                logger.info(f"Generated: {generated_text[:100]}...")
                logger.info(f"BLEU-4: {bleu_4:.4f}, METEOR: {meteor:.4f}, CIDEr: {cider:.4f}")
                logger.info(f"TTFT: {ttft:.3f}s, Total Time: {total_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Error processing sample {sample.get('file_id', i)}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Calculate aggregate metrics
    if all_metrics:
        logger.info("Calculating aggregate metrics...")
        
        # Calculate corpus-level and sentence-level metrics
        import sacrebleu
        hyps = [r['generated_text'] for r in results]
        refs_list = [r['reference_caption'] for r in results]
        
        # Corpus-level BLEU-1 and BLEU-4
        # Select tokenization based on language
        # Chinese: 'zh' - character-level tokenization
        # English: '13a' - standard English tokenization
        bleu_tokenize = 'zh' if language == 'chinese' else '13a'
        
        # Compute corpus-level BLEU using sacrebleu
        # This returns a BLEUScore object with all BLEU-n scores
        corpus_bleu_result = sacrebleu.corpus_bleu(hyps, [refs_list], tokenize=bleu_tokenize)
        
        # Extract BLEU-4 (the default score)
        overall_bleu_4 = corpus_bleu_result.score / 100.0  # Convert to [0,1] range
        
        # Extract BLEU-1 from precisions (precisions[0] is BLEU-1)
        # precisions is a list of [BLEU-1, BLEU-2, BLEU-3, BLEU-4] as percentages
        overall_bleu_1 = corpus_bleu_result.precisions[0] / 100.0  # Convert to [0,1] range
        
        # Corpus-level METEOR, ROUGE-L, CIDEr
        corpus_meteor = metrics.compute_corpus_meteor(hyps, refs_list, language=language)
        corpus_rouge_l = metrics.compute_corpus_rouge_l(hyps, refs_list)
        corpus_cider = metrics.compute_corpus_cider(hyps, refs_list, language=language)
        
        # Average sentence-level metrics
        avg_bleu_1_sentence = np.mean([r['bleu_1_score'] for r in results])
        avg_bleu_4_sentence = np.mean([r['bleu_4_score'] for r in results])
        avg_meteor_sentence = np.mean([r['meteor_score'] for r in results])
        avg_rouge_l_sentence = np.mean([r['rouge_l_score'] for r in results])
        avg_cider = np.mean([r['cider_score'] for r in results])
        avg_bertscore_precision = np.mean([r['bertscore_precision'] for r in results])
        avg_bertscore_recall = np.mean([r['bertscore_recall'] for r in results])
        avg_bertscore_f1 = np.mean([r['bertscore_f1'] for r in results])
        
        # Extract detailed latency metrics
        detailed_latency_data = [r['latency_metrics'] for r in results if r['latency_metrics']]
        latency_metrics = metrics.compute_detailed_latency_metrics(detailed_latency_data)
        
        overall_results = {
            "experiment_config": {
                "dataset_type": "unified",
                "dataset_path": manifest_path,
                "caption_type": caption_type,
                "language": language,
                "prompt_type": subtitle_flag,
                "max_samples": max_samples,
                "total_samples": len(results),
                "model": "Qwen-Audio"
            },
            "metrics": {
                # Corpus-level metrics
                f"corpus_bleu_1_{language[:2]}": overall_bleu_1,
                f"corpus_bleu_4_{language[:2]}": overall_bleu_4,
                "corpus_meteor": corpus_meteor,
                "corpus_rouge_l": corpus_rouge_l,
                "corpus_cider": corpus_cider,
                # Sentence-level metrics (averaged)
                "avg_bleu_1_sentence": avg_bleu_1_sentence,
                "avg_bleu_4_sentence": avg_bleu_4_sentence,
                "avg_meteor_sentence": avg_meteor_sentence,
                "avg_rouge_l_sentence": avg_rouge_l_sentence,
                "avg_cider": avg_cider,
                # BERTScore
                "avg_bertscore_precision": avg_bertscore_precision,
                "avg_bertscore_recall": avg_bertscore_recall,
                "avg_bertscore_f1": avg_bertscore_f1,
                # Latency metrics
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
        lang_display = "Chinese" if language == 'chinese' else "English"
        
        # Log corpus-level metrics
        logger.info("=" * 80)
        logger.info("CORPUS-LEVEL METRICS:")
        logger.info("=" * 80)
        logger.info(f"Corpus BLEU-1 ({lang_display} tokenization): {overall_bleu_1:.4f}")
        logger.info(f"Corpus BLEU-4 ({lang_display} tokenization): {overall_bleu_4:.4f}")
        logger.info(f"Corpus METEOR: {corpus_meteor:.4f}")
        logger.info(f"Corpus ROUGE-L: {corpus_rouge_l:.4f}")
        logger.info(f"Corpus CIDEr: {corpus_cider:.4f}")
        
        # Log sentence-level metrics
        logger.info("=" * 80)
        logger.info("SENTENCE-LEVEL METRICS (AVERAGE):")
        logger.info("=" * 80)
        logger.info(f"Average BLEU-1: {avg_bleu_1_sentence:.4f}")
        logger.info(f"Average BLEU-4: {avg_bleu_4_sentence:.4f}")
        logger.info(f"Average METEOR: {avg_meteor_sentence:.4f}")
        logger.info(f"Average ROUGE-L: {avg_rouge_l_sentence:.4f}")
        logger.info(f"Average CIDEr: {avg_cider:.4f}")
        
        # Log BERTScore
        logger.info("=" * 80)
        logger.info("BERTScore (AVERAGE):")
        logger.info("=" * 80)
        logger.info(f"Average BERTScore Precision: {avg_bertscore_precision:.4f}")
        logger.info(f"Average BERTScore Recall: {avg_bertscore_recall:.4f}")
        logger.info(f"Average BERTScore F1: {avg_bertscore_f1:.4f}")
        logger.info("=" * 80)
        
        # Log latency metrics
        logger.info("LATENCY & RESOURCE METRICS:")
        logger.info("=" * 80)
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
    parser = argparse.ArgumentParser(description="Run Qwen-Audio baseline experiment")
    parser.add_argument("--manifest_path", required=True, help="Path to manifest file")
    parser.add_argument("--model_path", required=True, help="Path to Qwen-Audio model")
    parser.add_argument("--output_name", default="qwen_audio_results", help="Output file name")
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples to process")
    parser.add_argument("--verbose", action="store_true", help="Print detailed results")
    parser.add_argument("--caption_type", default="audio_only", choices=["original", "audio_only"],
                       help="Type of caption to use")
    parser.add_argument("--language", default="english", choices=["chinese", "english"],
                       help="Language for generation")
    parser.add_argument("--subtitle_flag", default="subtitle", choices=["subtitle", "nosubtitle"],
                       help="Whether to use subtitle")
    parser.add_argument("--cache_dir", default="/data/gpfs/projects/punim2341/jiajunlu/hf-cache",
                       help="HuggingFace cache directory")
    
    args = parser.parse_args()
    
    run_qwen_audio_experiment(
        manifest_path=args.manifest_path,
        model_path=args.model_path,
        output_name=args.output_name,
        max_samples=args.max_samples,
        verbose=args.verbose,
        caption_type=args.caption_type,
        language=args.language,
        subtitle_flag=args.subtitle_flag,
        cache_dir=args.cache_dir
    )

if __name__ == "__main__":
    main()
