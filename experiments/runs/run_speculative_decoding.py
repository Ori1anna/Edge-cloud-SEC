#!/usr/bin/env python3
"""
Speculative Decoding Experiment Runner
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
from src.models.edge_model import EdgeModel
from src.models.cloud_model import CloudModel
from src.speculative_decoding import SimpleSpeculativeDecoding
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
#             return """根据音频，仅输出一条中文情感描述。
# 要求：只一行，句号结尾；不提问、不对话、不解释；不用引号、英文或数字。
# 内容只含“情感词 + 声学线索”（1–3 个情感 + 至少 1 条线索），用逗号或顿号连接。
# 情感词可用：悲伤、愤怒、开心、平静、焦虑、担忧、害怕、失望、嫉妒、冷淡、厌倦、轻蔑、真诚、自信、激动等；
# 声学线索可用：语调上扬、语调下降、语速加快、语速放慢、声音洪亮、声音低沉、声音颤抖、停顿明显、带哭腔、含笑、有气无力、铿锵有力、低声、压抑等。
# 即使线索较弱，也写出最可能的情感并附一条声学线索。"""
            return """你是“音频情感描述器”。只做一件事：依据音频的可听线索，输出一条简洁的中文情感描述。

【输出格式（必须严格遵守）】
- 只输出一行中文短句；不加引号；不用任何前缀（如“示例/编号/输出/结果”）；不使用英文或“视频/文本”等字样；不提问、不寒暄、不解释过程。
- 句式参考数据集中标注的写法：用“情感词 + 声学线索 + 态度倾向”的短语式表达，短语之间用逗号/顿号连接，必要时句号收尾。
- 长度建议 15–40 个汉字；最多给出 1–3 种核心情感，不要过长枚举。

【风格词汇与表达偏好】
- 情感词优先使用：悲伤/难过/委屈/愤怒/恼怒/不满/嫉妒/怨恨/开心/喜悦/兴奋/满足/幸福/惊讶/疑惑/好奇/焦虑/担忧/害怕/无奈/失望/平静/冷淡/自豪/骄傲/鄙视/轻蔑/尊重/真诚 等。
- 声学线索优先使用：语调高扬/下降、语速快/慢、音量高/低、声音颤抖、断断续续、带哭腔、含笑意、停顿明显、铿锵有力、轻快、低沉/尖锐 等。
- 生成风格贴近如下参考（仅作风格示意，勿原样照抄、勿输出“示例”二字）：伤心难过，声音颤抖，情绪激动失望。/ 嫉妒，不满，生出怨恨情绪。/ 心情喜悦无比，兴高采烈。/ 语气激动，语调愤怒，声音洪亮，怒不可遏。/ 声音断断续续，带哭腔，悲痛难抑。/ 语调高亢急促，铿锵有力，情绪激昂。/ 语调冷淡舒缓，情绪颓靡，不以为意。

【判别与约束】
- 严禁输出：“无法判断情绪”“请提供更多信息”“好的/嗯”等对话式内容。
- 如线索不充分，也要**就近选择最可能的情感**并给出至少一条声学线索；仅当音频为空/强噪声无法听清时，才允许输出：情绪不明显。
- 不捏造具体人物/事件/语义内容，只依据声音特征与可听到的情绪状态作描述。

【你的唯一任务】
- 从音频中提取可听到的情绪与相关声学线索，按上述格式输出一条中文描述。
- 只输出这“一行描述”，除此之外不要输出任何其他文字。"""
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

def run_speculative_decoding_experiment(config_path: str = "configs/default.yaml", 
                                      dataset_type: str = "unified",
                                      dataset_path: str = None,
                                      output_name: str = "speculative_decoding_results",
                                      max_samples: int = None, 
                                      verbose: bool = False,
                                      caption_type: str = "original",
                                      language: str = "chinese",
                                      prompt_type: str = "default",
                                      entropy_threshold: float = 1.5,
                                      k: int = 3,
                                      gate_type: str = "entropy",
                                      margin_threshold: float = 0.1,
                                      logprob_threshold: float = -2.0):
    """
    Run speculative decoding experiment with flexible configuration
    
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
        entropy_threshold: Threshold for entropy-based uncertainty
        k: Number of draft tokens to generate
        gate_type: Gating strategy ("entropy", "margin", "logprob")
        margin_threshold: Threshold for top-1 vs top-2 margin gating
        logprob_threshold: Threshold for token log-prob gating
    """
    
    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Load dataset
    if dataset_path is None:
        raise ValueError("dataset_path must be specified")
    
    data = load_dataset(dataset_type, dataset_path)
    if max_samples:
        data = data[:max_samples]
    
    logger.info(f"Processing {len(data)} samples")
    logger.info(f"Caption type: {caption_type}")
    logger.info(f"Language: {language}")
    logger.info(f"Prompt type: {prompt_type}")
    logger.info(f"Speculative Decoding params: entropy_threshold={entropy_threshold}, k={k}, gate_type={gate_type}, margin_threshold={margin_threshold}, logprob_threshold={logprob_threshold}")
    
    # Initialize models and processors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    audio_processor = AudioProcessor(**config['audio'])
    edge_model = EdgeModel(**config['models']['edge'])
    cloud_model = CloudModel(**config['models']['cloud'])
    
    # Initialize speculative decoding system
    spec_decoding = SimpleSpeculativeDecoding(
        edge_model=edge_model,
        cloud_model=cloud_model,
        entropy_threshold=entropy_threshold,
        k=k,
        gate_type=gate_type,
        margin_threshold=margin_threshold,
        logprob_threshold=logprob_threshold
    )
    
    metrics = EvaluationMetrics()
    
    # Get prompt template
    prompt_template = get_prompt_template(prompt_type, language)
    logger.info(f"Using prompt template: {prompt_template}")
    
    # Run experiment
    results = []
    total_latency = 0
    
    for i, sample in enumerate(data):
        logger.info(f"Processing sample {i+1}/{len(data)}: {sample['file_id']}")
        
        try:
            # Load audio
            audio_path = sample['audio_path']
            if not os.path.exists(audio_path):
                logger.warning(f"Audio file not found: {audio_path}")
                continue
            
            audio_waveform = audio_processor.load_audio(audio_path)
            
            # Get reference caption
            reference_caption = get_caption_field(sample, caption_type, language)
            if not reference_caption:
                logger.warning(f"No reference caption found for {sample['file_id']}")
                continue
            
            # Generate text using speculative decoding
            generated_text, latency_metrics = spec_decoding.generate(
                audio_features=audio_waveform,
                prompt=prompt_template,
                max_new_tokens=64  # Increased to allow for longer Chinese emotion descriptions
            )
            
            # Calculate metrics
            bleu_score = metrics.compute_bleu([reference_caption], generated_text)
            cider_score = metrics.compute_cider([reference_caption], generated_text)
            
            # Extract speculative decoding specific metrics from latency_metrics
            spec_metrics = {
                'total_time': latency_metrics.get('total_time', 0),
                'output_tokens': latency_metrics.get('output_tokens', 0),
                'cloud_calls': latency_metrics.get('cloud_calls', 0),
                'total_draft_tokens': latency_metrics.get('total_draft_tokens', 0),
                'total_accepted_tokens': latency_metrics.get('total_accepted_tokens', 0),
                'total_corrections': latency_metrics.get('total_corrections', 0),
                'cloud_verified_tokens': latency_metrics.get('cloud_verified_tokens', 0),
                'acceptance_rate': latency_metrics.get('acceptance_rate', 0),
                'correction_rate': latency_metrics.get('correction_rate', 0),
                'cloud_call_rate': latency_metrics.get('cloud_call_rate', 0),
                'token_level_cloud_rate': latency_metrics.get('token_level_cloud_rate', 0),
                'corrected_token_rate': latency_metrics.get('corrected_token_rate', 0)
            }
            
            # Store results
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
                "latency_metrics": latency_metrics,
                "speculative_decoding_metrics": spec_metrics,
                "speculative_decoding_config": {
                    "entropy_threshold": entropy_threshold,
                    "k": k
                }
            }
            results.append(result)
            
            if verbose:
                logger.info(f"Sample {i+1}:")
                logger.info(f"  Reference: {reference_caption}")
                logger.info(f"  Generated: {generated_text}")
                logger.info(f"  BLEU: {bleu_score:.4f}")
                logger.info(f"  CIDEr: {cider_score:.4f}")
                
                # Show latency metrics (compatible with baseline format)
                latency_metrics = result['latency_metrics']
                logger.info(f"  TTFT: {latency_metrics.get('ttft', 0):.4f}s")
                logger.info(f"  ITPS: {latency_metrics.get('itps', 0):.2f} tokens/sec")
                logger.info(f"  OTPS: {latency_metrics.get('otps', 0):.2f} tokens/sec")
                logger.info(f"  OET: {latency_metrics.get('oet', 0):.4f}s")
                logger.info(f"  Total Time: {latency_metrics.get('total_time', 0):.4f}s")
                logger.info(f"  CPU: {latency_metrics.get('cpu_percent', 0):.1f}%")
                logger.info(f"  RAM: {latency_metrics.get('ram_gb', 0):.2f}GB")
                logger.info(f"  GPU: {latency_metrics.get('gpu_util', 0):.1f}%")
                logger.info(f"  GPU Memory: {latency_metrics.get('gpu_memory_gb', 0):.2f}GB")
                
                # Show speculative decoding specific metrics
                if spec_metrics:
                    logger.info(f"  Speculative Decoding Metrics:")
                    logger.info(f"    Cloud Calls: {spec_metrics.get('cloud_calls', 0)}")
                    logger.info(f"    Total Draft Tokens: {spec_metrics.get('total_draft_tokens', 0)}")
                    logger.info(f"    Total Accepted Tokens: {spec_metrics.get('total_accepted_tokens', 0)}")
                    logger.info(f"    Total Corrections: {spec_metrics.get('total_corrections', 0)}")
                    logger.info(f"    Acceptance Rate: {spec_metrics.get('acceptance_rate', 0):.2%}")
                    logger.info(f"    Correction Rate: {spec_metrics.get('correction_rate', 0):.2%}")
                    logger.info(f"    Cloud Call Rate: {spec_metrics.get('cloud_call_rate', 0):.2%}")
                    logger.info(f"    Token-level Cloud Rate: {spec_metrics.get('token_level_cloud_rate', 0):.2%}")
                    logger.info(f"    Corrected Token Rate: {spec_metrics.get('corrected_token_rate', 0):.2%}")
                logger.info("")
            
        except Exception as e:
            logger.error(f"Error processing sample {sample['file_id']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Calculate overall metrics
    if results:
        avg_bleu = sum(r['bleu_score'] for r in results) / len(results)
        avg_cider = sum(r['cider_score'] for r in results) / len(results)
        
        # Extract detailed latency metrics (compatible with baseline format)
        detailed_latency_data = [r['latency_metrics'] for r in results if r['latency_metrics']]
        latency_metrics = metrics.compute_detailed_latency_metrics(detailed_latency_data)
        
        # Extract speculative decoding metrics
        spec_metrics_data = [r['speculative_decoding_metrics'] for r in results if r['speculative_decoding_metrics']]
        
        # Calculate aggregated speculative decoding metrics
        if spec_metrics_data:
            total_time_mean = sum(m.get('total_time', 0) for m in spec_metrics_data) / len(spec_metrics_data)
            avg_cloud_calls = sum(m.get('cloud_calls', 0) for m in spec_metrics_data) / len(spec_metrics_data)
            avg_acceptance_rate = sum(m.get('acceptance_rate', 0) for m in spec_metrics_data) / len(spec_metrics_data)
            avg_correction_rate = sum(m.get('correction_rate', 0) for m in spec_metrics_data) / len(spec_metrics_data)
            avg_cloud_call_rate = sum(m.get('cloud_call_rate', 0) for m in spec_metrics_data) / len(spec_metrics_data)
            avg_token_level_cloud_rate = sum(m.get('token_level_cloud_rate', 0) for m in spec_metrics_data) / len(spec_metrics_data)
            avg_corrected_token_rate = sum(m.get('corrected_token_rate', 0) for m in spec_metrics_data) / len(spec_metrics_data)
            
            aggregated_spec_metrics = {
                "total_time_mean": total_time_mean,
                "avg_cloud_calls": avg_cloud_calls,
                "avg_acceptance_rate": avg_acceptance_rate,
                "avg_correction_rate": avg_correction_rate,
                "avg_cloud_call_rate": avg_cloud_call_rate,
                "avg_token_level_cloud_rate": avg_token_level_cloud_rate,
                "avg_corrected_token_rate": avg_corrected_token_rate,
                "total_samples": len(spec_metrics_data)
            }
        else:
            aggregated_spec_metrics = {}
        
        overall_results = {
            "experiment_config": {
                "dataset_type": dataset_type,
                "dataset_path": dataset_path,
                "caption_type": caption_type,
                "language": language,
                "prompt_type": prompt_type,
                "max_samples": max_samples,
                "total_samples": len(results),
                "speculative_decoding_config": {
                    "entropy_threshold": entropy_threshold,
                    "k": k
                }
            },
            "metrics": {
                "avg_bleu": avg_bleu,
                "avg_cider": avg_cider,
                "latency_metrics": latency_metrics,
                "speculative_decoding_metrics": aggregated_spec_metrics
            },
            "detailed_results": results
        }
        
        # Save results
        output_file = f"experiments/results/{output_name}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(overall_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to: {output_file}")
        logger.info(f"Average BLEU: {avg_bleu:.4f}")
        logger.info(f"Average CIDEr: {avg_cider:.4f}")
        
        # Log latency metrics (compatible with baseline format)
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
        
        # Log speculative decoding metrics
        if aggregated_spec_metrics:
            logger.info("Speculative Decoding Metrics:")
            logger.info(f"  Average Total Time: {aggregated_spec_metrics.get('total_time_mean', 0):.4f}s")
            logger.info(f"  Average Cloud Calls: {aggregated_spec_metrics.get('avg_cloud_calls', 0):.2f}")
            logger.info(f"  Average Acceptance Rate: {aggregated_spec_metrics.get('avg_acceptance_rate', 0):.2%}")
            logger.info(f"  Average Correction Rate: {aggregated_spec_metrics.get('avg_correction_rate', 0):.2%}")
            logger.info(f"  Average Cloud Call Rate: {aggregated_spec_metrics.get('avg_cloud_call_rate', 0):.2%}")
            logger.info(f"  Average Token-level Cloud Rate: {aggregated_spec_metrics.get('avg_token_level_cloud_rate', 0):.2%}")
            logger.info(f"  Average Corrected Token Rate: {aggregated_spec_metrics.get('avg_corrected_token_rate', 0):.2%}")
            logger.info(f"  Total Samples: {aggregated_spec_metrics.get('total_samples', 0)}")
        
        return overall_results
    else:
        logger.error("No results generated")
        return None

def main():
    parser = argparse.ArgumentParser(description="Run speculative decoding experiment")
    parser.add_argument("--config", default="configs/default.yaml", help="Configuration file path")
    parser.add_argument("--dataset_type", default="unified", help="Dataset type")
    parser.add_argument("--dataset_path", required=True, help="Path to dataset manifest file")
    parser.add_argument("--output_name", default="speculative_decoding_results", help="Output file name")
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples to process")
    parser.add_argument("--verbose", action="store_true", help="Print detailed results")
    parser.add_argument("--caption_type", default="original", choices=["original", "audio_only"], 
                       help="Type of caption to use")
    parser.add_argument("--language", default="chinese", choices=["chinese", "english"], 
                       help="Language for generation")
    parser.add_argument("--prompt_type", default="default", choices=["default", "detailed", "concise"], 
                       help="Type of prompt to use")
    
    # Speculative decoding specific parameters
    parser.add_argument("--entropy_threshold", type=float, default=1.5, 
                       help="Threshold for entropy-based uncertainty")
    parser.add_argument("--k", type=int, default=3, 
                       help="Number of draft tokens to generate")
    parser.add_argument("--gate_type", default="entropy", choices=["entropy", "margin", "logprob"],
                       help="Gating strategy: entropy, margin, or logprob")
    parser.add_argument("--margin_threshold", type=float, default=0.1,
                       help="Top-1 vs Top-2 margin threshold for margin gating")
    parser.add_argument("--logprob_threshold", type=float, default=-2.0,
                       help="Token log-prob threshold for logprob gating")
    
    args = parser.parse_args()
    
    run_speculative_decoding_experiment(
        config_path=args.config,
        dataset_type=args.dataset_type,
        dataset_path=args.dataset_path,
        output_name=args.output_name,
        max_samples=args.max_samples,
        verbose=args.verbose,
        caption_type=args.caption_type,
        language=args.language,
        prompt_type=args.prompt_type,
        entropy_threshold=args.entropy_threshold,
        k=args.k,
        gate_type=args.gate_type,
        margin_threshold=args.margin_threshold,
        logprob_threshold=args.logprob_threshold
    )

if __name__ == "__main__":
    main()
