#!/usr/bin/env python3
"""
CPU-Limited Edge-Only Baseline Experiment Runner
Simulates iPhone 15 Plus hardware constraints for realistic edge-cloud comparison
"""

import argparse
import json
import logging
import os
import sys
import time
import threading
import psutil
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import pandas as pd
import yaml

from src.data.audio_processor import AudioProcessor
from src.models.edge_model import EdgeModel
from src.evaluation.metrics import EvaluationMetrics

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HardwareLimiter:
    """Simulate iPhone 15 Plus hardware constraints"""
    
    def __init__(self, 
                 max_cpu_cores: int = 2,  # Limit to 2 performance cores
                 max_memory_gb: float = 6.0,  # Reserve 2GB for system
                 max_memory_mb: int = None):
        """
        Initialize hardware limiter
        
        Args:
            max_cpu_cores: Maximum CPU cores to use
            max_memory_gb: Maximum memory in GB
            max_memory_mb: Alternative memory limit in MB
        """
        self.max_cpu_cores = max_cpu_cores
        self.max_memory_gb = max_memory_gb
        self.max_memory_mb = max_memory_mb or int(max_memory_gb * 1024)
        
        # Original system info
        self.original_cpu_count = psutil.cpu_count()
        self.original_memory = psutil.virtual_memory().total
        
        logger.info(f"Hardware Limiter initialized:")
        logger.info(f"  Original CPU cores: {self.original_cpu_count}")
        logger.info(f"  Original memory: {self.original_memory / (1024**3):.1f}GB")
        logger.info(f"  Limited CPU cores: {self.max_cpu_cores}")
        logger.info(f"  Limited memory: {self.max_memory_gb:.1f}GB")
    
    def apply_limits(self):
        """Apply hardware limits to current process"""
        try:
            # Set CPU affinity to limit cores
            if hasattr(psutil, 'cpu_count') and psutil.cpu_count() > self.max_cpu_cores:
                # Get current process
                process = psutil.Process()
                
                # Set CPU affinity to first N cores
                available_cores = list(range(min(self.max_cpu_cores, psutil.cpu_count())))
                process.cpu_affinity(available_cores)
                
                logger.info(f"CPU affinity set to cores: {available_cores}")
            
            # Set memory limit using resource module
            import resource
            memory_limit_bytes = self.max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
            
            logger.info(f"Memory limit set to {self.max_memory_gb:.1f}GB")
            
        except Exception as e:
            logger.warning(f"Could not apply hardware limits: {e}")
            logger.warning("Continuing without hardware limits...")
    
    def monitor_usage(self) -> Dict[str, float]:
        """Monitor current resource usage"""
        try:
            process = psutil.Process()
            
            # CPU usage - need to call twice to get accurate reading
            cpu_percent = process.cpu_percent()
            if cpu_percent == 0.0:
                # If first call returns 0, try with a small interval
                cpu_percent = process.cpu_percent(interval=0.1)
            
            # Fallback to system CPU usage if process CPU is still 0
            if cpu_percent == 0.0:
                system_cpu = psutil.cpu_percent(interval=0.1)
                cpu_percent = min(system_cpu * 0.1, 100.0)  # Estimate process usage as 10% of system
            
            # Memory usage
            memory_info = process.memory_info()
            memory_gb = memory_info.rss / (1024**3)
            
            # System memory usage
            system_memory = psutil.virtual_memory()
            system_memory_gb = system_memory.used / (1024**3)
            
            return {
                'cpu_percent': max(cpu_percent, 0.1),  # Ensure minimum 0.1% to avoid 0 values
                'memory_gb': memory_gb,
                'system_memory_gb': system_memory_gb,
                'memory_percent': system_memory.percent
            }
            
        except Exception as e:
            logger.warning(f"Could not monitor resource usage: {e}")
            return {
                'cpu_percent': 0.0,
                'memory_gb': 0.0,
                'system_memory_gb': 0.0,
                'memory_percent': 0.0
            }

class LimitedEdgeModel(EdgeModel):
    """Edge model with hardware limitations"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Omni-3B", device: str = "cpu", 
                 name: str = None, dtype: str = "float32", hardware_limiter: HardwareLimiter = None):
        """
        Initialize limited edge model
        
        Args:
            model_name: Model name
            device: Device to use (forced to CPU for edge simulation)
            name: Alternative name parameter
            dtype: Data type (use float32 for CPU)
            hardware_limiter: Hardware limiter instance
        """
        # Force CPU for edge simulation
        device = "cpu"
        dtype = "float32"  # Use float32 for CPU
        
        self.hardware_limiter = hardware_limiter
        
        logger.info(f"Initializing LimitedEdgeModel with device={device}, dtype={dtype}")
        
        # Initialize parent class (model loading without strict memory limits)
        super().__init__(model_name=model_name, device=device, name=name, dtype=dtype)
        
        # Note: Memory limits are not applied after model loading to avoid instability
        
        # Set CPU-specific optimizations
        if device == "cpu":
            self._apply_cpu_optimizations()
    
    def _apply_cpu_optimizations(self):
        """Apply CPU-specific optimizations"""
        try:
            # Set number of threads for CPU inference
            torch.set_num_threads(self.hardware_limiter.max_cpu_cores if self.hardware_limiter else 2)
            
            # Enable CPU optimizations
            torch.backends.mkldnn.enabled = True
            torch.backends.mkl.enabled = True
            
            logger.info(f"Applied CPU optimizations with {torch.get_num_threads()} threads")
            
        except Exception as e:
            logger.warning(f"Could not apply CPU optimizations: {e}")
    
    def generate_draft(self, 
                      audio_features: torch.Tensor,
                      prompt: str = "Based on this audio, describe the emotional state of the speaker in Chinese.",
                      max_new_tokens: int = 32,
                      temperature: float = 0.7,
                      top_p: float = 0.9,
                      use_streaming: bool = True) -> tuple[str, dict]:
        """
        Generate draft text with hardware monitoring
        
        Args:
            audio_features: Audio waveform tensor
            prompt: Text prompt for generation
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            use_streaming: Whether to use streaming for accurate TTFT measurement
            
        Returns:
            Tuple of (generated_text, latency_metrics)
        """
        # Enable streaming for accurate TTFT measurement even on CPU
        # Streaming is essential for correct TTFT calculation
        
        # Start hardware monitoring
        monitor_data = {
            'cpu_samples': [], 
            'memory_samples': [],
            'gpu_util_samples': [],
            'gpu_memory_gb': 0.0
        }
        monitor_thread = None
        
        if self.hardware_limiter:
            def monitor_hardware():
                """Monitor hardware usage during generation"""
                start_time = time.time()
                while time.time() - start_time < 30:  # Monitor for up to 30 seconds
                    usage = self.hardware_limiter.monitor_usage()
                    monitor_data['cpu_samples'].append(usage['cpu_percent'])
                    monitor_data['memory_samples'].append(usage['memory_gb'])
                    
                    # Monitor GPU usage even on CPU-limited edge (for baseline comparison)
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        device_count = pynvml.nvmlDeviceGetCount()
                        if device_count > 0:
                            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                            gpu_util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            gpu_memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            monitor_data['gpu_util_samples'].append(gpu_util_info.gpu)
                            monitor_data['gpu_memory_gb'] = gpu_memory_info.used / (1024 ** 3)
                    except Exception:
                        # GPU monitoring not available, use default values
                        monitor_data['gpu_util_samples'].append(0.0)
                        monitor_data['gpu_memory_gb'] = 0.0
                    
                    time.sleep(0.1)  # Sample every 100ms
            
            monitor_thread = threading.Thread(target=monitor_hardware)
            monitor_thread.daemon = True
            monitor_thread.start()
        
        try:
            # Call parent method
            generated_text, latency_metrics = super().generate_draft(
                audio_features=audio_features,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                use_streaming=use_streaming
            )
            
            # Add hardware monitoring data to metrics
            if monitor_data['cpu_samples']:
                latency_metrics['cpu_avg'] = sum(monitor_data['cpu_samples']) / len(monitor_data['cpu_samples'])
                latency_metrics['cpu_max'] = max(monitor_data['cpu_samples'])
                latency_metrics['memory_avg_gb'] = sum(monitor_data['memory_samples']) / len(monitor_data['memory_samples'])
                latency_metrics['memory_max_gb'] = max(monitor_data['memory_samples'])
                
                # Add GPU monitoring data
                if monitor_data['gpu_util_samples']:
                    latency_metrics['gpu_util'] = max(monitor_data['gpu_util_samples'])
                    latency_metrics['gpu_memory_gb'] = monitor_data['gpu_memory_gb']
                else:
                    latency_metrics['gpu_util'] = 0.0
                    latency_metrics['gpu_memory_gb'] = 0.0
                
                # Add CPU and RAM usage (use average values for consistency with metrics calculation)
                latency_metrics['cpu_percent'] = latency_metrics['cpu_avg']  # Use average instead of last sample
                latency_metrics['ram_gb'] = latency_metrics['memory_avg_gb']  # Use average instead of last sample
            
            # Add hardware limit info
            if self.hardware_limiter:
                latency_metrics['hardware_limits'] = {
                    'max_cpu_cores': self.hardware_limiter.max_cpu_cores,
                    'max_memory_gb': self.hardware_limiter.max_memory_gb
                }
            
            return generated_text, latency_metrics
            
        except Exception as e:
            logger.error(f"Error in limited edge model generation: {e}")
            return "", {}
        
        finally:
            # Stop monitoring
            if monitor_thread and monitor_thread.is_alive():
                monitor_thread.join(timeout=0.5)

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
            return """任务：请基于给定音频，输出一句“情感说明短句”。

必须遵守：
- 只输出一句中文短句（12–30个汉字），以“。”结尾。
- 句子中同时包含：一个主要情绪 + 一个简短的声学/韵律线索（如语气、语速、强弱、音高变化等“类别”层面的描述即可），但不要解释或列举。
- 不要出现客套话、邀请继续对话、表情符号、英文、Markdown、标号或代码；不要提及“音频/模型/分析/我”。
- 若存在多种可能性，只选择最可能的一种，不要并列罗列。

只给出最终这“一句短句”，不要输出其他内容。"""

        elif language == "english":
            return "Please generate a concise English emotion description based on the audio content. Example: 'The speaker's voice trembles, expressing sadness and disappointment.'"
    elif prompt_type == "detailed":
        if language == "chinese":
            return """任务：请生成“情感说明长句”，按以下顺序组织内容并保持自然流畅：

(1) 先用2–3个“类别级”的声学/韵律线索描述说话方式（从以下维度中任选若干：语速、音调高低/起伏、音量强弱、停顿与连贯度、音色/紧张度等），不用给数值；
(2) 据此给出最可能的单一情绪（不列举选项）；
(3) 若语义内容暗示缘由，可用极简的一小短语点到为止（可用“可能/似乎/大概”表不确定）。

输出要求：
- 只输出“两到三句中文长句”，约70–100个字；
- 使用第三人称或“说话人”等指代；不要出现第一/第二人称；不要设问或邀请对话；
- 不要编造具体人物/时间/地点等细节；不要出现表情符号、英文、Markdown/代码。"""
        elif language == "english":
            # return "Please provide a detailed analysis of emotional features in the audio, including tone, speed, volume, etc., and generate a detailed English emotion description."
            return "As an expert in the field of emotions, please focus on the acoustic information in the audio to discern clues related to the emotions of the individual. Please provide a detailed description and ultimately predict the emotional state of the individual."
    elif prompt_type == "concise":
        if language == "chinese":
            return "请用最简洁的中文描述音频中的情感状态。"
        elif language == "english":
            return "Please describe the emotional state in the audio using the most concise English."
    else:
        raise ValueError(f"Unsupported prompt type: {prompt_type}")

def run_cpu_limited_edge_experiment(config_path: str = "configs/default.yaml", 
                                  dataset_type: str = "unified",
                                  dataset_path: str = None,
                                  output_name: str = "edge_cpu_limited_results",
                                  max_samples: int = None, 
                                  verbose: bool = False,
                                  caption_type: str = "original",
                                  language: str = "chinese",
                                  prompt_type: str = "default",
                                  input_modality: str = "audio_only",
                                  max_cpu_cores: int = 2,
                                  max_memory_gb: float = 6.0):
    """
    Run CPU-limited edge-only baseline experiment
    
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
        max_cpu_cores: Maximum CPU cores to use (iPhone 15 Plus: 2 performance cores)
        max_memory_gb: Maximum memory in GB (iPhone 15 Plus: 8GB total, reserve 2GB)
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
    
    logger.info(f"Processing {len(data)} samples with CPU-limited edge model")
    logger.info(f"Caption type: {caption_type}")
    logger.info(f"Language: {language}")
    logger.info(f"Prompt type: {prompt_type}")
    logger.info(f"Hardware limits: {max_cpu_cores} cores, {max_memory_gb}GB memory")
    
    # Initialize hardware limiter
    hardware_limiter = HardwareLimiter(
        max_cpu_cores=max_cpu_cores,
        max_memory_gb=max_memory_gb
    )
    
    # Initialize models and processors
    device = torch.device("cpu")  # Force CPU
    logger.info(f"Using device: {device}")
    
    audio_processor = AudioProcessor(**config['audio'])
    
    # Use limited edge model
    edge_model = LimitedEdgeModel(
        model_name=config['models']['edge']['name'],  # Use 'name' from config
        device="cpu",
        dtype="float32",  # Use float32 for CPU
        hardware_limiter=hardware_limiter
    )
    
    metrics = EvaluationMetrics()
    
    # Get prompt template
    prompt_template = get_prompt_template(prompt_type, language)
    logger.info(f"Using prompt template: {prompt_template[:100]}...")
    
    # Run experiment
    results = []
    
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
            
            # Generate using Speculative Decoding Edge logic for alignment
            # This ensures Edge Baseline uses the SAME generation logic as Speculative Decoding's Edge
            # Key changes from original generate_draft():
            #   - Uses custom incremental generation (not HF generate())
            #   - CJK-aware repetition penalty (1.22, content only)
            #   - Content-only 3-gram ban (removes punctuation)
            #   - Hard punctuation gate (4 chars for comma, 5 for period)
            #   - Same-character blocking for CJK
            #   - Fallback mechanism (top-k non-punctuation)
            #   - Stopping criteria (2 sentences, 90 chars, 48 tokens minimum)
            logger.info(f"Using Speculative Decoding Edge logic for sample {sample['file_id']}")
            generated_text, detailed_latency = edge_model.generate_draft_with_spec_logic(
                audio_waveform, 
                prompt_template, 
                max_new_tokens=128,         # Increased from 64 to 128 (match spec decoding)
                target_sentences=2,         # Same as spec decoding
                min_chars=90,               # Same as spec decoding
                min_new_tokens_sc=48,       # Same as spec decoding
                prompt_type=prompt_type     # Pass prompt type for stopping criteria
            )
            
            # Calculate traditional metrics
            bleu_1_score = metrics.compute_bleu_1([reference_caption], generated_text)
            bleu_4_score = metrics.compute_bleu_4([reference_caption], generated_text)
            meteor_score = metrics.compute_meteor([reference_caption], generated_text, language=language)
            rouge_l_score = metrics.compute_rouge_l([reference_caption], generated_text)
            cider_score = metrics.compute_cider([reference_caption], generated_text)
            
            # Store results (BERTScore will be calculated in batch later)
            result = {
                "file_id": sample['file_id'],
                "dataset": sample['dataset'],
                "reference_caption": reference_caption,
                "generated_text": generated_text,
                "bleu_1_score": bleu_1_score,
                "bleu_4_score": bleu_4_score,
                "meteor_score": meteor_score,
                "rouge_l_score": rouge_l_score,
                "cider_score": cider_score,
                "caption_type": caption_type,
                "language": language,
                "prompt_type": prompt_type,
                "latency_metrics": detailed_latency,
                "hardware_config": {
                    "device": "cpu",
                    "max_cpu_cores": max_cpu_cores,
                    "max_memory_gb": max_memory_gb,
                    "model_dtype": "float32"
                }
            }
            results.append(result)
            
            if verbose:
                logger.info(f"Sample {i+1}:")
                logger.info(f"  Reference: {reference_caption}")
                logger.info(f"  Generated: {generated_text}")
                logger.info(f"  BLEU-1: {bleu_1_score:.4f}")
                logger.info(f"  BLEU-4: {bleu_4_score:.4f}")
                logger.info(f"  METEOR: {meteor_score:.4f}")
                logger.info(f"  ROUGE-L: {rouge_l_score:.4f}")
                logger.info(f"  CIDEr: {cider_score:.4f}")
                logger.info(f"  BERTScore: Computing in batch...")
                if detailed_latency:
                    logger.info(f"  TTFT: {detailed_latency.get('ttft', 0):.4f}s")
                    logger.info(f"  Total Time: {detailed_latency.get('total_time', 0):.4f}s")
                    logger.info(f"  CPU Usage: {detailed_latency.get('cpu_avg', 0):.1f}% (max: {detailed_latency.get('cpu_max', 0):.1f}%)")
                    logger.info(f"  Memory Usage: {detailed_latency.get('memory_avg_gb', 0):.2f}GB (max: {detailed_latency.get('memory_max_gb', 0):.2f}GB)")
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
        
        # Calculate corpus-level BLEU with language-specific tokenization
        import sacrebleu
        hyps = [r['generated_text'] for r in results]
        refs = [[r['reference_caption'] for r in results]]
        
        # Select tokenization based on language
        # Chinese: 'zh' - character-level tokenization
        # English: '13a' - standard English tokenization (handles punctuation, case, etc.)
        bleu_tokenize = 'zh' if language == 'chinese' else '13a'
        corpus_bleu = sacrebleu.corpus_bleu(hyps, refs, tokenize=bleu_tokenize)
        overall_bleu = corpus_bleu.score / 100.0  # Convert to [0,1] range
        
        # Keep sentence-level averages for diagnostic purposes
        avg_bleu_1_sentence = sum(r['bleu_1_score'] for r in results) / len(results)
        avg_bleu_4_sentence = sum(r['bleu_4_score'] for r in results) / len(results)
        avg_meteor_sentence = sum(r['meteor_score'] for r in results) / len(results)
        avg_rouge_l_sentence = sum(r['rouge_l_score'] for r in results) / len(results)
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
                "total_samples": len(results),
                "hardware_config": {
                    "device": "cpu",
                    "max_cpu_cores": max_cpu_cores,
                    "max_memory_gb": max_memory_gb,
                    "model_dtype": "float32"
                }
            },
            "metrics": {
                f"corpus_bleu_{language[:2]}": overall_bleu,  # Language-specific BLEU (corpus-level)
                "avg_bleu_1_sentence": avg_bleu_1_sentence,  # Sentence-level BLEU-1 average
                "avg_bleu_4_sentence": avg_bleu_4_sentence,  # Sentence-level BLEU-4 average
                "avg_meteor_sentence": avg_meteor_sentence,  # Sentence-level METEOR average
                "avg_rouge_l_sentence": avg_rouge_l_sentence,  # Sentence-level ROUGE-L average
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
        lang_display = "Chinese" if language == 'chinese' else "English"
        logger.info(f"Corpus BLEU ({lang_display} tokenization): {overall_bleu:.4f}")
        logger.info(f"Average BLEU-1 (sentence-level): {avg_bleu_1_sentence:.4f}")
        logger.info(f"Average BLEU-4 (sentence-level): {avg_bleu_4_sentence:.4f}")
        logger.info(f"Average METEOR (sentence-level): {avg_meteor_sentence:.4f}")
        logger.info(f"Average ROUGE-L (sentence-level): {avg_rouge_l_sentence:.4f}")
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
            
            # Additional token statistics
            logger.info("Token Statistics:")
            logger.info(f"  Avg Input Tokens: {latency_metrics.get('avg_input_tokens', 0):.1f}")
            logger.info(f"  Avg Output Tokens: {latency_metrics.get('avg_output_tokens', 0):.1f}")
            logger.info(f"  Total Input Tokens: {latency_metrics.get('total_input_tokens', 0)}")
            logger.info(f"  Total Output Tokens: {latency_metrics.get('total_output_tokens', 0)}")
            
            # Hardware limits info
            logger.info("Hardware Configuration:")
            logger.info(f"  Device: CPU (limited)")
            logger.info(f"  Max CPU Cores: {hardware_limiter.max_cpu_cores}")
            logger.info(f"  Max Memory: {hardware_limiter.max_memory_gb:.1f}GB")
        
        return overall_results
    else:
        logger.error("No results generated")
        return None

def main():
    parser = argparse.ArgumentParser(description="Run CPU-limited edge-only baseline experiment")
    parser.add_argument("--config", default="configs/default.yaml", help="Configuration file path")
    parser.add_argument("--dataset_type", default="unified", help="Dataset type")
    parser.add_argument("--dataset_path", required=True, help="Path to dataset manifest file")
    parser.add_argument("--output_name", default="edge_cpu_limited_results", help="Output file name")
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
    parser.add_argument("--max_cpu_cores", type=int, default=2, 
                       help="Maximum CPU cores to use (iPhone 15 Plus: 2 performance cores)")
    parser.add_argument("--max_memory_gb", type=float, default=16.0, 
                       help="Maximum memory in GB (default: 16.0, increased for Qwen2.5-Omni-3B model)")
    
    args = parser.parse_args()
    
    run_cpu_limited_edge_experiment(
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
        max_cpu_cores=args.max_cpu_cores,
        max_memory_gb=args.max_memory_gb
    )

if __name__ == "__main__":
    main()
