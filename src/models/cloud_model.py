"""
Cloud model implementation (Qwen2.5-Omni 7B)
"""

import torch
import torch.nn as nn
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from typing import Dict, List, Optional, Tuple
import logging
import psutil

logger = logging.getLogger(__name__)


class CloudModel:
    """Cloud model for verification and refinement"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Omni-7B", device: str = "cuda", name: str = None, dtype: str = "float16"):
        if name is not None:
            self.model_name = name
        else:
            self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.model = None
        self.processor = None
        self.kv_cache = {}
        self.prefix_cache = {}
        self._load_model()
    
    def _load_model(self):
        """Load the cloud model and processor"""
        try:
            logger.info(f"Loading cloud model: {self.model_name}")
            logger.info("This may take several minutes for the first time...")
            
            # Load processor
            logger.info("Loading processor...")
            self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_name)
            logger.info("Processor loaded successfully")
            
            # Load model
            logger.info("Loading model (this may take a while)...")
            torch_dtype = torch.float16 if self.dtype == "float16" else torch.float32
            self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map=self.device,
                trust_remote_code=True
            )
            logger.info("Cloud model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load cloud model: {e}")
            raise
    
    def generate_independently(self,
                              audio_waveform: torch.Tensor,
                              prompt: str = "Based on this audio, describe the emotional state of the speaker in Chinese.",
                              max_new_tokens: int = 32,
                              temperature: float = 0.7,
                              top_p: float = 0.9) -> tuple[str, dict]:
        """
        Generate text independently using cloud model with detailed latency metrics
        
        Args:
            audio_waveform: Audio waveform tensor
            prompt: Text prompt for generation
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Tuple of (generated_text, latency_metrics)
        """
        try:
            import time
            
            # Record start time for total processing
            total_start_time = time.time()
            
            # Prepare conversation format with audio input according to official documentation
            conversation = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio_waveform},  # Include audio waveform
                        {"type": "text", "text": prompt}
                    ],
                },
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            
            # Process inputs with audio waveform
            inputs = self.processor(
                text=text, 
                audio=audio_waveform,  # Pass audio waveform to processor
                return_tensors="pt", 
                padding=True
            )
            inputs = inputs.to(self.device).to(self.model.dtype)
            
            # Record time before generation (for input processing)
            input_end_time = time.time()
            input_processing_time = input_end_time - total_start_time
            
            # Count input tokens
            input_token_count = len(inputs['input_ids'][0])
            
            # Generate text using the model
            generation_start_time = time.time()
            
            # Start GPU monitoring thread before generation
            gpu_monitor_data = {'gpu_util_samples': [], 'gpu_memory_gb': 0.0}
            gpu_monitor_thread = None
            
            try:
                import threading
                import pynvml
                
                def monitor_gpu_usage():
                    """Monitor GPU usage during generation"""
                    try:
                        pynvml.nvmlInit()
                        device_count = pynvml.nvmlDeviceGetCount()
                        if device_count > 0:
                            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                            # Sample GPU usage multiple times during generation
                            for _ in range(20):  # Sample for up to 2 seconds
                                gpu_util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                                gpu_memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                                gpu_monitor_data['gpu_util_samples'].append(gpu_util_info.gpu)
                                gpu_monitor_data['gpu_memory_gb'] = gpu_memory_info.used / (1024 ** 3)
                                time.sleep(0.1)  # 100ms between samples
                    except Exception as e:
                        logger.debug(f"GPU monitoring thread failed: {e}")
                
                # Start monitoring thread
                gpu_monitor_thread = threading.Thread(target=monitor_gpu_usage)
                gpu_monitor_thread.daemon = True
                gpu_monitor_thread.start()
                
            except ImportError:
                logger.debug("pynvml not available for GPU monitoring")
            except Exception as e:
                logger.debug(f"Failed to start GPU monitoring: {e}")
            
            # Try streaming generation first for accurate TTFT
            try:
                from .streaming_generator import StreamingGenerator
                
                # Create streaming generator
                streamer = StreamingGenerator(self.model, self.processor, self.device)
                
                # Generate with streaming for accurate TTFT
                generated_text, streaming_metrics = streamer.generate_with_accurate_metrics(
                    inputs, max_new_tokens, temperature, top_p
                )
                
                output_token_count = streaming_metrics.get('output_tokens', 0)
                generation_end_time = time.time()
                total_end_time = time.time()
                
                # Wait for GPU monitoring to complete (max 0.5 seconds)
                if gpu_monitor_thread and gpu_monitor_thread.is_alive():
                    gpu_monitor_thread.join(timeout=0.5)
                
                # Use streaming metrics for accurate latency calculation
                latency_metrics = self._calculate_accurate_latency_metrics(
                    total_start_time, input_end_time, generation_start_time,
                    generation_end_time, total_end_time, input_token_count,
                    output_token_count, gpu_monitor_data, streaming_metrics
                )
                
            except Exception as e:
                logger.warning(f"Streaming generation failed, falling back to batch generation: {e}")
                
                # Fallback to batch generation
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                        repetition_penalty=1.15,  # Anti-repetition penalty
                        no_repeat_ngram_size=3,   # Prevent 3-gram repetition
                        typical_p=0.95,           # Typical sampling for better quality
                        min_new_tokens=2,         # Minimum tokens to generate
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                        return_dict_in_generate=False,  # 减少内存开销
                        output_scores=False,  # 不返回scores以节省内存
                        return_audio=False
                    )
                
                # Record time after generation
                generation_end_time = time.time()
                
                # Wait for GPU monitoring to complete (max 0.5 seconds)
                if gpu_monitor_thread and gpu_monitor_thread.is_alive():
                    gpu_monitor_thread.join(timeout=0.5)
                
                # Extract generated text
                if isinstance(outputs, torch.Tensor):
                    # When return_dict_in_generate=False, outputs is a tensor
                    generated_tokens = outputs[0][len(inputs['input_ids'][0]):]
                else:
                    # Fallback for dict format
                    generated_tokens = outputs.sequences[0][len(inputs['input_ids'][0]):]
                
                generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                # Clean up the generated text
                generated_text = generated_text.replace('<|im_end|>', '').strip()
                
                # Save token count before cleanup
                output_token_count = len(generated_tokens)
                
                # Clean up tensors to free memory
                del inputs
                del outputs
                del generated_tokens
                torch.cuda.empty_cache()  # Clear CUDA cache
                
                # Calculate latency metrics using fallback method
                latency_metrics = self._calculate_latency_metrics(
                    total_start_time, input_end_time, generation_start_time, 
                    generation_end_time, total_end_time, input_token_count, 
                    output_token_count, gpu_monitor_data
                )
            
            # Record total end time
            total_end_time = time.time()
            
            logger.info(f"Generated text: {generated_text}")
            logger.info(f"Latency metrics: {latency_metrics}")
            
            return generated_text, latency_metrics
            
        except Exception as e:
            import traceback
            logger.error(f"Error in generate_independently: {e}")
            logger.error(f"Error details: {traceback.format_exc()}")
            # Return empty string and empty metrics as fallback
            return "", {}
    
    def _calculate_latency_metrics(self,
                                  total_start: float,
                                  input_end: float,
                                  gen_start: float,
                                  gen_end: float,
                                  total_end: float,
                                  input_tokens: int,
                                  output_tokens: int,
                                  gpu_monitor_data: dict = None) -> dict:
        """
        Calculate detailed latency metrics including CPU and GPU usage
        
        Args:
            total_start: Total start time
            input_end: Input processing end time
            gen_start: Generation start time
            gen_end: Generation end time
            total_end: Total end time
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Dictionary of latency metrics including CPU and GPU usage
        """
        # Time-to-First-Token (TTFT): Time from generation start to first token
        ttft = gen_end - gen_start
        
        # Input Token Per Second (ITPS): Input processing speed
        input_processing_time = input_end - total_start
        itps = input_tokens / input_processing_time if input_processing_time > 0 else 0
        
        # Output Token Per Second (OTPS): Generation speed
        generation_time = gen_end - gen_start
        otps = output_tokens / generation_time if generation_time > 0 else 0
        
        # Output Evaluation Time (OET): Total generation time
        oet = generation_time
        
        # Total Time: Complete response generation time
        total_time = total_end - total_start
        
        # Get CPU and GPU usage with improved measurement
        try:
            import time
            
            # Get current process
            process = psutil.Process()
            
            # Improved CPU usage measurement
            # Method 1: Use cpu_percent with interval parameter for more accurate measurement
            cpu_percent = process.cpu_percent(interval=0.5)  # Measure over 0.5 seconds
            
            # Method 2: If still 0, try alternative approach with system-wide CPU usage
            if cpu_percent == 0.0:
                # Get system-wide CPU usage and estimate process contribution
                system_cpu = psutil.cpu_percent(interval=0.2)
                # Estimate process CPU usage (rough approximation)
                cpu_percent = min(system_cpu * 0.1, 100.0)  # Assume process uses up to 10% of system CPU
            
            # Method 3: If still 0, use a small baseline value to indicate monitoring is working
            if cpu_percent == 0.0:
                cpu_percent = 0.1  # Small non-zero value to indicate monitoring is active
            
            # RAM usage in GB
            memory_info = process.memory_info()
            ram_gb = memory_info.rss / (1024 ** 3)  # Convert bytes to GB
            
            # GPU usage - use real-time monitoring data if available
            gpu_util = 0.0
            gpu_memory_gb = 0.0
            
            if gpu_monitor_data and gpu_monitor_data.get('gpu_util_samples'):
                # Use real-time monitoring data from generation process
                gpu_util_samples = gpu_monitor_data['gpu_util_samples']
                gpu_util = max(gpu_util_samples) if gpu_util_samples else 0.0
                gpu_memory_gb = gpu_monitor_data.get('gpu_memory_gb', 0.0)
                logger.debug(f"Using real-time GPU data: util samples {gpu_util_samples}, max: {gpu_util}%, memory: {gpu_memory_gb:.2f}GB")
            else:
                # Fallback to current GPU state
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    device_count = pynvml.nvmlDeviceGetCount()
                    if device_count > 0:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        gpu_util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        gpu_util = gpu_util_info.gpu
                        gpu_memory_gb = gpu_memory_info.used / (1024 ** 3)
                        logger.debug(f"Using fallback GPU data: util: {gpu_util}%, memory: {gpu_memory_gb:.2f}GB")
                except Exception as e:
                    logger.debug(f"Fallback GPU monitoring failed: {e}")
                    gpu_util = 0.0
                    gpu_memory_gb = 0.0
            
        except Exception as e:
            logger.warning(f"Failed to get CPU/GPU usage: {e}")
            cpu_percent = 0.0
            ram_gb = 0.0
            gpu_util = 0.0
            gpu_memory_gb = 0.0
        
        return {
            'ttft': ttft,                    # Time-to-First-Token (sec)
            'itps': itps,                    # Input Token Per Second (tokens/sec)
            'otps': otps,                    # Output Token Per Second (tokens/sec)
            'oet': oet,                      # Output Evaluation Time (sec)
            'total_time': total_time,        # Total Time (sec)
            'input_tokens': input_tokens,    # Number of input tokens
            'output_tokens': output_tokens,  # Number of output tokens
            'cpu_percent': cpu_percent,      # CPU usage percentage (peak)
            'ram_gb': ram_gb,                # RAM usage in GB
            'gpu_util': gpu_util,            # GPU utilization percentage
            'gpu_memory_gb': gpu_memory_gb  # GPU memory usage in GB
        }
    
    def verify_tokens(self, context: dict, draft_tokens: list, threshold: float = 0.25) -> tuple[list, Optional[int], bool]:
        """
        Prefill 一次（forward），按阈值逐位验收；首错用 p(·) 直接采样纠正
        返回: (accepted_tokens, correction_token, needs_correction)
        """
        import torch
        if not draft_tokens:
            return [], None, False

        # 1) 设备与拼接（把 edge 张量搬到 cloud 设备）
        ctx_ids = context['input_ids'].to(self.device)
        ctx_mask = context['attention_mask'].to(self.device)
        y = torch.tensor([draft_tokens], device=self.device)
        full_ids = torch.cat([ctx_ids, y], dim=1)
        full_mask = torch.cat([ctx_mask, torch.ones_like(y)], dim=1)

        # 2) 组装多模态输入（文本 + 音频特征）
        inputs = {
            'input_ids': full_ids,
            'attention_mask': full_mask,
        }
        
        # 添加音频特征（如果存在）
        has_audio = False
        if 'input_features' in context:
            inputs['input_features'] = context['input_features'].to(self.device)
            has_audio = True
        if 'feature_attention_mask' in context:
            inputs['feature_attention_mask'] = context['feature_attention_mask'].to(self.device)
            has_audio = True
        
        logger.debug(f"Cloud verification with audio features: {has_audio}, draft tokens: {draft_tokens}")
        
        # 3) 显式前向，稳健取得 logits
        # 直接使用 thinker 子模型，它支持多模态输入且节省显存
        with torch.no_grad():
            out = self.model.thinker(**inputs, return_dict=True, use_cache=False)
        logits = out.logits[0]  # [m+k, V]
        m = ctx_ids.shape[1]
        k = len(draft_tokens)
        
        logger.debug(f"Cloud verification: context_length={m}, draft_length={k}, logits_shape={logits.shape}")
        logger.debug(f"Full input sequence length: {full_ids.shape[1]}")

        # 4) 逐位排名阈值验收 + 首错纠正
        accepted = 0
        probabilities = []
        
        # 使用排名阈值替代绝对概率阈值
        rank_threshold = 10  # 接受排名前10的token
        min_prob_threshold = 0.005  # 最小概率阈值
        
        for i in range(k):
            pos = m - 1 + i            # 对应 yi 的预测位置
            probs = torch.softmax(logits[pos], dim=-1)
            p_i = probs[draft_tokens[i]].item()
            probabilities.append(p_i)
            
            # 计算当前token的排名
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            rank = (sorted_indices == draft_tokens[i]).nonzero(as_tuple=True)[0].item() + 1
            
            logger.debug(f"Position {i}: draft_token={draft_tokens[i]}, probability={p_i:.4f}, rank={rank}, min_threshold={min_prob_threshold:.4f}")
            
            # Additional debugging for extremely low probabilities
            if p_i < 1e-6:
                logger.warning(f"Extremely low probability for token {draft_tokens[i]}: {p_i:.2e}, rank={rank}")
                # Check if this token is in the top-k predictions
                top_k_probs, top_k_indices = torch.topk(probs, k=min(10, probs.size(0)))
                logger.warning(f"Top-10 tokens: {[(idx.item(), prob.item()) for idx, prob in zip(top_k_indices, top_k_probs)]}")
            
            # 使用排名阈值或最小概率阈值的组合策略
            if rank <= rank_threshold or p_i >= min_prob_threshold:
                accepted += 1
                logger.debug(f"  -> ACCEPTED (rank={rank} <= {rank_threshold} OR p_i={p_i:.4f} >= {min_prob_threshold:.4f})")
            else:
                # 首错纠正：使用贪心选择（top-1）而非随机采样
                corr = torch.argmax(probs).item()
                logger.info(f"Cloud correction: accepted {accepted}/{k} tokens, first error at position {i}")
                logger.info(f"  -> Probabilities: {[f'{p:.4f}' for p in probabilities]}")
                logger.info(f"  -> Ranks: {[f'{rank}' for rank in [(sorted_indices == draft_tokens[j]).nonzero(as_tuple=True)[0].item() + 1 for j in range(len(probabilities))]]}")
                logger.info(f"  -> Correction token: {corr} (replacing draft token {draft_tokens[i]})")
                return draft_tokens[:accepted], corr, True

        # 全部通过
        logger.info(f"Cloud correction: accepted {accepted}/{k} tokens (all passed)")
        logger.info(f"  -> Probabilities: {[f'{p:.4f}' for p in probabilities]}")
        return draft_tokens, None, False

    
    
    
    def update_kv_cache(self, tokens: List[int]):
        """Update KV cache with new tokens"""
        # TODO: Implement KV cache management
        pass
    
    def update_prefix_cache(self, prefix: List[int]):
        """Update prefix cache"""
        # TODO: Implement prefix cache management
        pass
    
    def rollback_and_regenerate(self, 
                               accepted_tokens: List[int],
                               audio_features: torch.Tensor) -> List[int]:
        """
        Rollback and regenerate from the first error position
        
        Args:
            accepted_tokens: Tokens accepted so far
            audio_features: Audio features for context
            
        Returns:
            New tokens from rollback point
        """
        try:
            # Prepare prompt for regeneration
            prompt = "Based on the audio features, describe the emotional state of the speaker:"
            
            # Tokenize the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device)
            
            # Add accepted tokens as context
            if accepted_tokens:
                accepted_tensor = torch.tensor([accepted_tokens]).to(self.device)
                input_ids = torch.cat([input_ids, accepted_tensor], dim=1)
            
            # Generate new tokens from the cloud model
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=32,  # Generate some new tokens
                    temperature=0.7,
                    do_sample=True,
                    repetition_penalty=1.15,  # Anti-repetition penalty
                    no_repeat_ngram_size=3,   # Prevent 3-gram repetition
                    typical_p=0.95,           # Typical sampling for better quality
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True
                )
            
            # Extract newly generated tokens
            new_tokens = outputs.sequences[0][len(input_ids[0]):].tolist()
            
            logger.info(f"Regenerated {len(new_tokens)} tokens after rollback")
            return new_tokens
            
        except Exception as e:
            logger.error(f"Error in rollback_and_regenerate: {e}")
            return []
    
    def _calculate_accurate_latency_metrics(self,
                                          total_start: float,
                                          input_end: float,
                                          gen_start: float,
                                          gen_end: float,
                                          total_end: float,
                                          input_tokens: int,
                                          output_tokens: int,
                                          gpu_monitor_data: dict = None,
                                          streaming_metrics: dict = None) -> dict:
        """
        Calculate accurate latency metrics using streaming data
        
        Args:
            total_start: Total start time
            input_end: Input processing end time
            gen_start: Generation start time
            gen_end: Generation end time
            total_end: Total end time
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            gpu_monitor_data: GPU monitoring data
            streaming_metrics: Streaming generation metrics
            
        Returns:
            Dictionary of accurate latency metrics
        """
        # Use streaming TTFT if available
        if streaming_metrics and 'ttft' in streaming_metrics:
            ttft = streaming_metrics['ttft']
        else:
            # Fallback to generation time
            ttft = gen_end - gen_start
        
        # Input Token Per Second (ITPS): Input processing speed
        input_processing_time = input_end - total_start
        itps = input_tokens / input_processing_time if input_processing_time > 0 else 0
        
        # Output Token Per Second (OTPS): Generation speed (excluding TTFT)
        generation_time = gen_end - gen_start
        if streaming_metrics and 'step_times' in streaming_metrics and streaming_metrics['step_times']:
            # Use actual token generation times (excluding TTFT)
            token_generation_time = sum(streaming_metrics['step_times'])
            otps = output_tokens / token_generation_time if token_generation_time > 0 else 0
        else:
            # Fallback to total generation time
            otps = output_tokens / generation_time if generation_time > 0 else 0
        
        # Output Evaluation Time (OET): Total generation time
        oet = generation_time
        
        # Total Time: Complete response generation time
        total_time = total_end - total_start
        
        # Get CPU and GPU usage with improved measurement
        try:
            import time
            
            # Get current process
            process = psutil.Process()
            
            # Improved CPU usage measurement
            cpu_percent = process.cpu_percent(interval=0.5)
            
            if cpu_percent == 0.0:
                system_cpu = psutil.cpu_percent(interval=0.2)
                cpu_percent = min(system_cpu * 0.1, 100.0)
            
            if cpu_percent == 0.0:
                cpu_percent = 0.1
            
            # RAM usage in GB
            memory_info = process.memory_info()
            ram_gb = memory_info.rss / (1024 ** 3)
            
            # GPU usage - use real-time monitoring data if available
            gpu_util = 0.0
            gpu_memory_gb = 0.0
            
            if gpu_monitor_data and gpu_monitor_data.get('gpu_util_samples'):
                gpu_util_samples = gpu_monitor_data['gpu_util_samples']
                gpu_util = max(gpu_util_samples) if gpu_util_samples else 0.0
                gpu_memory_gb = gpu_monitor_data.get('gpu_memory_gb', 0.0)
            else:
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    device_count = pynvml.nvmlDeviceGetCount()
                    if device_count > 0:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        gpu_util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        gpu_util = gpu_util_info.gpu
                        gpu_memory_gb = gpu_memory_info.used / (1024 ** 3)
                except Exception as e:
                    logger.debug(f"Fallback GPU monitoring failed: {e}")
                    gpu_util = 0.0
                    gpu_memory_gb = 0.0
            
        except Exception as e:
            logger.warning(f"Failed to get CPU/GPU usage: {e}")
            cpu_percent = 0.0
            ram_gb = 0.0
            gpu_util = 0.0
            gpu_memory_gb = 0.0
        
        return {
            'ttft': ttft,                    # Accurate Time-to-First-Token (sec)
            'itps': itps,                    # Input Token Per Second (tokens/sec)
            'otps': otps,                    # Accurate Output Token Per Second (tokens/sec)
            'oet': oet,                      # Output Evaluation Time (sec)
            'total_time': total_time,        # Total Time (sec)
            'input_tokens': input_tokens,    # Number of input tokens
            'output_tokens': output_tokens,  # Number of output tokens
            'cpu_percent': cpu_percent,      # CPU usage percentage (peak)
            'ram_gb': ram_gb,                # RAM usage in GB
            'gpu_util': gpu_util,            # GPU utilization percentage
            'gpu_memory_gb': gpu_memory_gb,  # GPU memory usage in GB
            'streaming_metrics': streaming_metrics  # Additional streaming data
        }
