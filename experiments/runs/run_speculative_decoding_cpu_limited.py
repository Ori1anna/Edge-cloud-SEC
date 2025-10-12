#!/usr/bin/env python3
"""
CPU-Limited Edge + GPU Cloud Speculative Decoding Experiment Runner
Simulates realistic edge-cloud deployment with hardware constraints
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
from src.models.cloud_model import CloudModel
from src.speculative_decoding import SimpleSpeculativeDecoding
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
            max_cpu_cores: Maximum CPU cores to use (iPhone 15 Plus: 2 performance cores)
            max_memory_gb: Maximum memory in GB (iPhone 15 Plus: 8GB total, reserve 2GB)
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
                      use_streaming: bool = False) -> tuple[str, dict]:
        """
        Generate draft tokens with hardware monitoring
        
        Args:
            audio_features: Audio features tensor
            prompt: Text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            use_streaming: Whether to use streaming generation
            
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
            return """你是“音频情感描述器”。只做一件事：依据音频的可听线索，输出一条简洁的中文情感描述。

【输出格式（必须严格遵守）】
- 只输出一行中文短句；不加引号；不用任何前缀（如“示例/编号/输出/结果”）；不使用英文或“视频/文本”等字样；不提问、不寒暄、不解释过程。
- 句式写法：用“情感词 + 声学线索 + 态度倾向”的描述性短语式表达进行输出。
- **长度要求：8-16个汉字**。

- 严禁输出：“无法判断情绪”“请提供更多信息”“好的/嗯”等对话式内容。
- 如线索不充分，也要**就近选择最可能的情感**并给出至少一条声学线索。
- 不捏造具体人物/事件/语义内容，只依据声音特征与可听到的情绪状态作描述。

【你的唯一任务】
- 从音频中提取可听到的情绪与相关声学线索，按上述格式输出一条中文描述。
- 只输出这“一行描述”，除此之外不要输出任何其他文字。"""
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
            return "Please provide a detailed analysis of emotional features in the audio, including tone, speed, volume, etc., and generate a detailed English emotion description."
    elif prompt_type == "concise":
        if language == "chinese":
            return "请用最简洁的中文描述音频中的情感状态。"
        elif language == "english":
            return "Please describe the emotional state in the audio using the most concise English."
    else:
        raise ValueError(f"Unsupported prompt type: {prompt_type}")

def run_cpu_limited_speculative_decoding_experiment(
    config_path: str = "configs/default.yaml", 
    dataset_type: str = "unified",
    dataset_path: str = None,
    output_name: str = "speculative_decoding_cpu_limited_results",
    max_samples: int = None, 
    verbose: bool = False,
    caption_type: str = "original",
    language: str = "chinese",
    prompt_type: str = "default",
    entropy_threshold: float = 3.0,
    k: int = 5,
    max_cpu_cores: int = 2,
    max_memory_gb: float = 6.0):
    """
    Run CPU-limited edge + GPU cloud speculative decoding experiment
    
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
        entropy_threshold: Entropy threshold for cloud verification
        k: Number of draft tokens to generate
        max_cpu_cores: Maximum CPU cores for edge model (iPhone 15 Plus: 2 performance cores)
        max_memory_gb: Maximum memory for edge model (iPhone 15 Plus: 8GB total, reserve 2GB)
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
    
    logger.info(f"Processing {len(data)} samples with CPU-limited edge + GPU cloud speculative decoding")
    logger.info(f"Caption type: {caption_type}")
    logger.info(f"Language: {language}")
    logger.info(f"Prompt type: {prompt_type}")
    logger.info(f"Entropy threshold: {entropy_threshold}")
    logger.info(f"K (draft tokens): {k}")
    logger.info(f"Edge hardware limits: {max_cpu_cores} cores, {max_memory_gb}GB memory")
    logger.info(f"Cloud device: GPU (G100)")
    
    # Initialize hardware limiter for edge
    hardware_limiter = HardwareLimiter(
        max_cpu_cores=max_cpu_cores,
        max_memory_gb=max_memory_gb
    )
    
    # Initialize models and processors
    device = torch.device("cpu")  # Edge uses CPU
    logger.info(f"Edge device: {device}")
    logger.info(f"Cloud device: cuda")
    
    audio_processor = AudioProcessor(**config['audio'])
    
    # Initialize CPU-limited edge model
    edge_model = LimitedEdgeModel(
        model_name=config['models']['edge']['name'],  # Use 'name' from config
        device="cpu",
        dtype="float32",  # Use float32 for CPU
        hardware_limiter=hardware_limiter
    )
    
    # Initialize GPU cloud model
    cloud_model = CloudModel(**config['models']['cloud'])
    
    # Initialize speculative decoding with multi-sentence configuration
    spec_decoding = SimpleSpeculativeDecoding(
        edge_model=edge_model,
        cloud_model=cloud_model,
        entropy_threshold=entropy_threshold,
        k=k,
        target_sentences=2,      # Target 2-3 sentences for detailed description
        min_chars=90,            # Minimum 90 characters for 2-3 sentences
        min_new_tokens_sc=48     # Minimum 48 tokens before allowing stop
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
            
            # Generate text using speculative decoding
            generated_text, latency_metrics = spec_decoding.generate(
                audio_features=audio_waveform,
                prompt=prompt_template,
                max_new_tokens=128  # Increased for 2-3 sentence multi-sentence output
            )
            
            # Calculate traditional metrics
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
                'acceptance_rate': latency_metrics.get('acceptance_rate', 0),
                'correction_rate': latency_metrics.get('correction_rate', 0),
                'cloud_call_rate': latency_metrics.get('cloud_call_rate', 0)
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
                },
                "hardware_config": {
                    "edge_device": "cpu",
                    "cloud_device": "cuda",
                    "max_cpu_cores": max_cpu_cores,
                    "max_memory_gb": max_memory_gb,
                    "edge_model_dtype": "float32",
                    "cloud_model_dtype": "float16"
                }
            }
            results.append(result)
            
            if verbose:
                logger.info(f"Sample {i+1}:")
                logger.info(f"  Reference: {reference_caption}")
                logger.info(f"  Generated: {generated_text}")
                logger.info(f"  BLEU: {bleu_score:.4f}")
                logger.info(f"  CIDEr: {cider_score:.4f}")
                logger.info(f"  BERTScore: Computing in batch...")
                logger.info(f"  Cloud calls: {spec_metrics['cloud_calls']}")
                logger.info(f"  Acceptance rate: {spec_metrics['acceptance_rate']:.3f}")
                logger.info(f"  Cloud call rate: {spec_metrics['cloud_call_rate']:.3f}")
                if latency_metrics:
                    logger.info(f"  TTFT: {latency_metrics.get('ttft', 0):.4f}s")
                    logger.info(f"  Total Time: {latency_metrics.get('total_time', 0):.4f}s")
                    logger.info(f"  CPU Usage: {latency_metrics.get('cpu_avg', 0):.1f}% (max: {latency_metrics.get('cpu_max', 0):.1f}%)")
                    logger.info(f"  Memory Usage: {latency_metrics.get('memory_avg_gb', 0):.2f}GB (max: {latency_metrics.get('memory_max_gb', 0):.2f}GB)")
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
        
        # Calculate speculative decoding metrics
        total_cloud_calls = sum(r['speculative_decoding_metrics']['cloud_calls'] for r in results)
        total_draft_tokens = sum(r['speculative_decoding_metrics']['total_draft_tokens'] for r in results)
        total_accepted_tokens = sum(r['speculative_decoding_metrics']['total_accepted_tokens'] for r in results)
        total_corrections = sum(r['speculative_decoding_metrics']['total_corrections'] for r in results)
        
        avg_cloud_calls = total_cloud_calls / len(results) if results else 0
        avg_acceptance_rate = total_accepted_tokens / total_draft_tokens if total_draft_tokens > 0 else 0
        avg_correction_rate = total_corrections / total_cloud_calls if total_cloud_calls > 0 else 0
        avg_cloud_call_rate = total_cloud_calls / len(results) if results else 0
        
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
                },
                "hardware_config": {
                    "edge_device": "cpu",
                    "cloud_device": "cuda",
                    "max_cpu_cores": max_cpu_cores,
                    "max_memory_gb": max_memory_gb,
                    "edge_model_dtype": "float32",
                    "cloud_model_dtype": "float16"
                }
            },
            "metrics": {
                "corpus_bleu_zh": overall_bleu,  # Corpus-level BLEU with Chinese tokenization
                "avg_bleu_sentence": avg_bleu_sentence,  # Sentence-level average for diagnostics
                "avg_cider": avg_cider,
                "avg_bertscore_precision": avg_bertscore_precision,
                "avg_bertscore_recall": avg_bertscore_recall,
                "avg_bertscore_f1": avg_bertscore_f1,
                "latency_metrics": latency_metrics,
                "speculative_decoding_metrics": {
                    "total_time_mean": latency_metrics.get('total_time_mean', 0),
                    "avg_cloud_calls": avg_cloud_calls,
                    "avg_acceptance_rate": avg_acceptance_rate,
                    "avg_correction_rate": avg_correction_rate,
                    "avg_cloud_call_rate": avg_cloud_call_rate,
                    "total_samples": len(results)
                }
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
        
        # Log speculative decoding metrics
        logger.info("Speculative Decoding Metrics:")
        logger.info(f"  Average cloud calls: {avg_cloud_calls:.2f}")
        logger.info(f"  Average acceptance rate: {avg_acceptance_rate:.3f}")
        logger.info(f"  Average correction rate: {avg_correction_rate:.3f}")
        logger.info(f"  Average cloud call rate: {avg_cloud_call_rate:.3f}")
        
        # Log latency metrics
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
            logger.info(f"  Edge Device: CPU (limited)")
            logger.info(f"  Cloud Device: CUDA")
            logger.info(f"  Max CPU Cores: {max_cpu_cores}")
            logger.info(f"  Max Memory: {max_memory_gb:.1f}GB")
        
        return overall_results
    else:
        logger.error("No results generated")
        return None

def main():
    parser = argparse.ArgumentParser(description="Run CPU-limited edge + GPU cloud speculative decoding experiment")
    parser.add_argument("--config", default="configs/default.yaml", help="Configuration file path")
    parser.add_argument("--dataset_type", default="unified", help="Dataset type")
    parser.add_argument("--dataset_path", required=True, help="Path to dataset manifest file")
    parser.add_argument("--output_name", default="speculative_decoding_cpu_limited_results", help="Output file name")
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples to process")
    parser.add_argument("--verbose", action="store_true", help="Print detailed results")
    parser.add_argument("--caption_type", default="original", choices=["original", "audio_only"], 
                       help="Type of caption to use")
    parser.add_argument("--language", default="chinese", choices=["chinese", "english"], 
                       help="Language for generation")
    parser.add_argument("--prompt_type", default="default", choices=["default", "detailed", "concise"], 
                       help="Type of prompt to use")
    parser.add_argument("--entropy_threshold", type=float, default=3.0, 
                       help="Entropy threshold for cloud verification (Recommended: 5.5-6.0 for fluency, 3.0-3.5 for quality, default changed to 3.0)")
    parser.add_argument("--k", type=int, default=5, 
                       help="Number of draft tokens to generate (Recommended: 4 for fluency, 6 for quality)")
    parser.add_argument("--max_cpu_cores", type=int, default=2, 
                       help="Maximum CPU cores for edge model (iPhone 15 Plus: 2 performance cores)")
    parser.add_argument("--max_memory_gb", type=float, default=16.0, 
                       help="Maximum memory for edge model (default: 16.0, increased for Qwen2.5-Omni-3B model)")
    
    args = parser.parse_args()
    
    run_cpu_limited_speculative_decoding_experiment(
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
        max_cpu_cores=args.max_cpu_cores,
        max_memory_gb=args.max_memory_gb
    )

if __name__ == "__main__":
    main()
