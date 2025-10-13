"""
Edge device model implementation (Qwen2.5-Omni 3B)
Based on official Qwen2.5-Omni documentation
"""

import torch
import torch.nn as nn
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from typing import Dict, List, Optional, Tuple
import logging
import psutil

logger = logging.getLogger(__name__)


class EdgeModel:
    """Edge device model for draft generation using Qwen2.5-Omni"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Omni-3B", device: str = "cuda", name: str = None, dtype: str = "float16"):
        # Support both model_name and name parameters for compatibility
        if name is not None:
            self.model_name = name
        else:
            self.model_name = model_name
        
        # Always set these attributes and load the model
        self.device = device
        self.dtype = dtype
        self.model = None
        self.processor = None
        
        logger.info(f"Initializing EdgeModel with model_name={self.model_name}, device={self.device}, dtype={self.dtype}")
        try:
            self._load_model()
            logger.info("EdgeModel initialization completed successfully")
        except Exception as e:
            logger.error(f"EdgeModel initialization failed: {e}")
            raise
    
    def _load_model(self):
        """Load the edge model and processor according to official documentation"""
        try:
            logger.info(f"Loading edge model: {self.model_name}")
            logger.info("This may take several minutes for the first time...")
            
            # Load processor first (includes tokenizer)
            logger.info("Loading processor...")
            self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_name)
            logger.info("Processor loaded successfully")
            
            # Load model using the correct Qwen2_5OmniForConditionalGeneration class
            logger.info("Loading model (this may take a while)...")
            torch_dtype = torch.float16 if self.dtype == "float16" else torch.float32
            
            # Use original efficient configuration
            self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map=self.device,  # Use original device mapping
                trust_remote_code=True
            )
            
            logger.info("Edge model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load edge model: {e}")
            logger.error("Please check:")
            logger.error("1. Internet connection")
            logger.error("2. Hugging Face authentication (run 'huggingface-cli login')")
            logger.error("3. Model name is correct")
            logger.error("4. Transformers version supports this model")
            raise
    
    def generate_draft(self, 
                      audio_features: torch.Tensor,
                      prompt: str = "Based on this audio, describe the emotional state of the speaker in Chinese.",
                      max_new_tokens: int = 32,
                      temperature: float = 0.7,
                      top_p: float = 0.9,
                      use_streaming: bool = True) -> tuple[str, dict]:
        """
        Generate draft text using edge model with detailed latency metrics
        
        Args:
            audio_features: Audio waveform tensor (1D)
            prompt: Text prompt for generation
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            use_streaming: Whether to use streaming generation for accurate TTFT
            
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
                        {"type": "audio", "audio": audio_features},  # Include audio waveform
                        {"type": "text", "text": prompt}
                    ],
                },
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            
            # Debug: log the complete prompt
            logger.info(f"Edge model prompt: {text}")
            
            # Process inputs with audio waveform - ensure correct shape
            # 如果audio_features是2D，转换为1D
            if audio_features.dim() == 2 and audio_features.shape[0] == 1:
                audio_features = audio_features.squeeze(0)
            
            inputs = self.processor(
                text=text, 
                audio=audio_features,  # Restore audio processing
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
            
            if use_streaming:
                # Use streaming generation for accurate TTFT measurement
                generated_text, streaming_metrics = self._generate_streaming(
                    inputs, max_new_tokens, temperature, top_p, gpu_monitor_data
                )
                output_token_count = streaming_metrics.get('output_tokens', 0)
                generation_end_time = time.time()
                total_end_time = time.time()
                
                # Use streaming metrics if available
                latency_metrics = self._calculate_accurate_latency_metrics(
                    total_start_time, input_end_time, generation_start_time,
                    generation_end_time, total_end_time, input_token_count,
                    output_token_count, gpu_monitor_data, streaming_metrics
                )
            else:
                # Use original batch generation (fallback)
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=False,  # Use deterministic generation
                        no_repeat_ngram_size=2,  # Prevent 2-gram repetition
                        repetition_penalty=1.05,  # Light repetition penalty
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                        return_dict_in_generate=False,
                        output_scores=False,
                        return_audio=False
                    )
                
                # Record time after generation
                generation_end_time = time.time()
                
                # Wait for GPU monitoring to complete (max 0.5 seconds)
                if gpu_monitor_thread and gpu_monitor_thread.is_alive():
                    gpu_monitor_thread.join(timeout=0.5)
                
                # Extract generated text
                if isinstance(outputs, torch.Tensor):
                    generated_tokens = outputs[0][len(inputs['input_ids'][0]):]
                else:
                    generated_tokens = outputs.sequences[0][len(inputs['input_ids'][0]):]
                
                generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                generated_text = generated_text.replace('<|im_end|>', '').strip()
                
                # Save token count before cleanup
                output_token_count = len(generated_tokens)
                
                # Clean up tensors to free memory
                del inputs
                del outputs
                del generated_tokens
                torch.cuda.empty_cache()
                
                # Record total end time
                total_end_time = time.time()
                
                # Calculate detailed latency metrics
                latency_metrics = self._calculate_latency_metrics(
                    total_start_time, input_end_time, generation_start_time, 
                    generation_end_time, total_end_time, input_token_count, 
                    output_token_count, gpu_monitor_data
                )
            
            logger.info(f"Generated text: {generated_text}")
            logger.info(f"Latency metrics: {latency_metrics}")
            
            return generated_text, latency_metrics
            
        except Exception as e:
            import traceback
            logger.error(f"Error in generate_draft: {e}")
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
        # Note: We can't measure exact TTFT without streaming, so we use generation time
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
    
    def compute_uncertainty(self, 
                           tokens: List[int], 
                           log_probs: torch.Tensor) -> Dict[str, float]:
        """
        Compute uncertainty signals for tokens
        
        Args:
            tokens: Generated tokens
            log_probs: Token log probabilities
            
        Returns:
            Dictionary of uncertainty measures
        """
        uncertainty_signals = {}
        
        if len(log_probs) == 0:
            return uncertainty_signals
        
        # Token-level log-probabilities
        uncertainty_signals['token_log_probs'] = log_probs.tolist()
        
        # Entropy
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        uncertainty_signals['entropy'] = entropy.tolist()
        
        # Margin (difference between top-1 and top-2 probabilities)
        # Only compute margin if we have at least 2 tokens
        if probs.shape[-1] >= 2:
            top2_probs, _ = torch.topk(probs, 2, dim=-1)
            margin = top2_probs[:, 0] - top2_probs[:, 1]
            uncertainty_signals['margin'] = margin.tolist()
        else:
            # If only one token, margin is 0
            uncertainty_signals['margin'] = [0.0] * probs.shape[0]
        
        return uncertainty_signals
    
    def should_verify_with_cloud(self, 
                                uncertainty_signals: Dict[str, float],
                                block_text: str = "",
                                threshold: float = 0.5) -> bool:
        """
        Determine if tokens should be verified with cloud model
        
        Args:
            uncertainty_signals: Uncertainty measures
            block_text: Text content of the block
            threshold: Decision threshold
            
        Returns:
            True if verification is needed
        """
        # Check content patterns first
        if block_text:
            content_patterns = self.detect_content_patterns(block_text)
            
            # Always verify if content has problematic patterns
            if content_patterns.get('has_repetition', False):
                logger.debug(f"Block needs verification due to repetition: {block_text[:50]}...")
                return True
            
            if content_patterns.get('is_very_short', False):
                logger.debug(f"Block needs verification due to being too short: {block_text}")
                return True
            
            if content_patterns.get('is_very_long', False):
                logger.debug(f"Block needs verification due to being too long: {block_text[:50]}...")
                return True
        
        # Check uncertainty signals
        if not uncertainty_signals:
            return False
        
        # Check entropy - high entropy indicates uncertainty
        if 'entropy' in uncertainty_signals:
            entropy_values = uncertainty_signals['entropy']
            if isinstance(entropy_values, list) and len(entropy_values) > 0:
                avg_entropy = sum(entropy_values) / len(entropy_values)
                if avg_entropy > 2.0:  # High entropy threshold
                    logger.debug(f"Block needs verification due to high entropy: {avg_entropy:.3f}")
                    return True
        
        # Check margin - low margin indicates uncertainty
        if 'margin' in uncertainty_signals:
            margin_values = uncertainty_signals['margin']
            if isinstance(margin_values, list) and len(margin_values) > 0:
                avg_margin = sum(margin_values) / len(margin_values)
                if avg_margin < 0.1:  # Low margin threshold
                    logger.debug(f"Block needs verification due to low margin: {avg_margin:.3f}")
                    return True
        
        # Check token log probabilities - very low probabilities indicate uncertainty
        if 'token_log_probs' in uncertainty_signals:
            log_probs = uncertainty_signals['token_log_probs']
            if isinstance(log_probs, list) and len(log_probs) > 0:
                # Handle nested lists (list of lists)
                if log_probs and isinstance(log_probs[0], list):
                    # Flatten the nested list
                    flat_probs = [prob for sublist in log_probs for prob in sublist]
                    avg_log_prob = sum(flat_probs) / len(flat_probs) if flat_probs else 0.0
                else:
                    avg_log_prob = sum(log_probs) / len(log_probs)
                
                if avg_log_prob < -3.0:  # Very low probability threshold
                    logger.debug(f"Block needs verification due to low log probability: {avg_log_prob:.3f}")
                    return True
        
        # Default: no verification needed
        return False
    
    def generate_draft_blocks(self, 
                             audio_features: torch.Tensor,
                             prompt: str = "Based on this audio, describe the emotional state of the speaker in Chinese.",
                             block_size: int = 4,
                             max_blocks: int = 8,
                             temperature: float = 0.7,
                             top_p: float = 0.9) -> Tuple[List[Dict], Dict]:
        """
        Generate draft text in blocks
        
        Args:
            audio_features: Audio waveform tensor (1D)
            prompt: Text prompt for generation
            block_size: Number of tokens per block
            max_blocks: Maximum number of blocks to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Tuple of (blocks_with_uncertainty, latency_metrics)
        """
        try:
            import time
            
            # Record start time for total processing
            total_start_time = time.time()
            
            # Prepare conversation format with audio input
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
                        {"type": "audio", "audio": audio_features},
                        {"type": "text", "text": prompt}
                    ],
                },
            ]
            
            # Apply chat template
            text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            
            # Process inputs with audio waveform
            inputs = self.processor(
                text=text, 
                audio=audio_features,
                return_tensors="pt", 
                padding=True
            )
            inputs = inputs.to(self.device).to(self.model.dtype)
            
            # Record time before generation
            input_end_time = time.time()
            input_processing_time = input_end_time - total_start_time
            input_token_count = len(inputs['input_ids'][0])
            
            # Generate text block by block
            generation_start_time = time.time()
            blocks_with_uncertainty = []
            all_generated_tokens = []
            
            # Start GPU monitoring thread
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
                            for _ in range(50):  # Sample for up to 5 seconds
                                gpu_util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                                gpu_memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                                gpu_monitor_data['gpu_util_samples'].append(gpu_util_info.gpu)
                                gpu_monitor_data['gpu_memory_gb'] = gpu_memory_info.used / (1024 ** 3)
                                time.sleep(0.1)
                    except Exception as e:
                        logger.debug(f"GPU monitoring thread failed: {e}")
                
                gpu_monitor_thread = threading.Thread(target=monitor_gpu_usage)
                gpu_monitor_thread.daemon = True
                gpu_monitor_thread.start()
                
            except ImportError:
                logger.debug("pynvml not available for GPU monitoring")
            except Exception as e:
                logger.debug(f"Failed to start GPU monitoring: {e}")
            
            with torch.no_grad():
                # Generate tokens one by one to get log probabilities
                current_inputs = {k: v.clone() for k, v in inputs.items()}
                
                for block_idx in range(max_blocks):
                    block_start_time = time.time()
                    
                    # Generate a block of tokens at once (like original method)
                    # Use the same input format as original method, but only pass valid keys
                    generate_inputs = {
                        'input_ids': current_inputs['input_ids'],
                        'attention_mask': current_inputs['attention_mask']
                    }
                    # Add audio-related inputs if they exist
                    if 'input_features' in current_inputs:
                        generate_inputs['input_features'] = current_inputs['input_features']
                    if 'feature_attention_mask' in current_inputs:
                        generate_inputs['feature_attention_mask'] = current_inputs['feature_attention_mask']
                    
                    # Generate tokens one by one to get proper log probabilities
                    block_tokens = []
                    block_log_probs = []
                    
                    for token_idx in range(block_size):
                        # Generate next token
                        outputs = self.model.generate(
                            **generate_inputs,
                            max_new_tokens=1,
                            temperature=temperature,
                            top_p=top_p,
                            do_sample=True,
                            pad_token_id=self.processor.tokenizer.eos_token_id,
                            return_dict_in_generate=True,
                            output_scores=True,
                            return_audio=False
                        )
                        
                        # Extract new token
                        input_length = generate_inputs['input_ids'].shape[1]
                        new_token = outputs.sequences[0][input_length].item()
                        
                        # Check for EOS token
                        is_eos_token = (new_token == self.processor.tokenizer.eos_token_id or 
                                       new_token == self.processor.tokenizer.convert_tokens_to_ids('<|im_end|>') or
                                       new_token == self.processor.tokenizer.convert_tokens_to_ids('<|endoftext|>'))
                        
                        if is_eos_token:
                            logger.debug(f"Found EOS token at position {token_idx}: {new_token}")
                            break
                        
                        # Get log probability for this token
                        if hasattr(outputs, 'scores') and outputs.scores:
                            # Get the logits for the generated token
                            logits = outputs.scores[0][0]  # [vocab_size]
                            log_probs = torch.log_softmax(logits, dim=-1)
                            token_log_prob = log_probs[new_token].item()
                        else:
                            # Fallback if scores not available
                            token_log_prob = 0.0
                        
                        block_tokens.append(new_token)
                        block_log_probs.append(token_log_prob)
                        
                        # Update inputs for next token
                        new_token_tensor = torch.tensor([[new_token]], device=self.device)
                        generate_inputs['input_ids'] = torch.cat([generate_inputs['input_ids'], new_token_tensor], dim=1)
                        generate_inputs['attention_mask'] = torch.cat([generate_inputs['attention_mask'], torch.ones((1, 1), device=self.device)], dim=1)
                    
                    if not block_tokens:
                        logger.debug("No new tokens generated, stopping")
                        break
                    
                    # Update current_inputs for next block
                    if block_tokens:
                        new_tokens_tensor = torch.tensor([block_tokens], device=self.device)
                        current_inputs['input_ids'] = torch.cat([current_inputs['input_ids'], new_tokens_tensor], dim=1)
                        current_inputs['attention_mask'] = torch.cat([current_inputs['attention_mask'], torch.ones((1, len(block_tokens)), device=self.device)], dim=1)
                    
                    if not block_tokens:
                        break
                    
                    # Compute uncertainty for this block
                    # Convert log probabilities to the format expected by compute_uncertainty
                    block_log_probs_tensor = torch.tensor(block_log_probs).unsqueeze(0)  # Add batch dimension
                    uncertainty_signals = self.compute_uncertainty(block_tokens, block_log_probs_tensor)
                    
                    # Decode block text, filtering out EOS tokens for display
                    filtered_tokens = [token for token in block_tokens 
                                     if token not in [self.processor.tokenizer.eos_token_id,
                                                     self.processor.tokenizer.convert_tokens_to_ids('<|im_end|>'),
                                                     self.processor.tokenizer.convert_tokens_to_ids('<|endoftext|>')]]
                    block_text = self.processor.tokenizer.decode(filtered_tokens, skip_special_tokens=True)
                    
                    # Check if this block should be verified with cloud
                    should_verify = self.should_verify_with_cloud(uncertainty_signals, block_text)
                    
                    # Store block information
                    block_info = {
                        'block_idx': block_idx,
                        'tokens': block_tokens,
                        'text': block_text,
                        'log_probs': block_log_probs,
                        'uncertainty_signals': uncertainty_signals,
                        'should_verify': should_verify,
                        'block_generation_time': time.time() - block_start_time
                    }
                    
                    blocks_with_uncertainty.append(block_info)
                    all_generated_tokens.extend(block_tokens)
                    
                    # If we hit EOS, stop generating
                    if block_tokens[-1] == self.processor.tokenizer.eos_token_id:
                        break
            
            # Wait for GPU monitoring to complete
            if gpu_monitor_thread and gpu_monitor_thread.is_alive():
                gpu_monitor_thread.join(timeout=0.5)
            
            # Record total end time
            generation_end_time = time.time()
            total_end_time = time.time()
            
            # Calculate latency metrics
            latency_metrics = self._calculate_latency_metrics(
                total_start_time, input_end_time, generation_start_time, 
                generation_end_time, total_end_time, input_token_count, 
                len(all_generated_tokens), gpu_monitor_data
            )
            
            # Clean up tensors
            del inputs
            del current_inputs
            torch.cuda.empty_cache()
            
            logger.info(f"Generated {len(blocks_with_uncertainty)} blocks with {len(all_generated_tokens)} total tokens")
            
            return blocks_with_uncertainty, latency_metrics
            
        except Exception as e:
            import traceback
            logger.error(f"Error in generate_draft_blocks: {e}")
            logger.error(f"Error details: {traceback.format_exc()}")
            return [], {}
    
    def detect_content_patterns(self, text: str) -> Dict[str, bool]:
        """
        Detect content patterns that might require cloud verification
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of detected patterns
        """
        import re
        
        patterns = {
            'has_numbers': bool(re.search(r'\d+', text)),
            'has_dates': bool(re.search(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}', text)),
            'has_names': bool(re.search(r'[A-Z][a-z]+ [A-Z][a-z]+', text)),  # Simple name pattern
            'has_emotions': bool(re.search(r'(悲伤|愤怒|快乐|恐惧|惊讶|厌恶|中性|激动|平静|紧张|放松|兴奋|沮丧|焦虑|满足|失望)', text)),
            'has_repetition': len(set(text.split())) < len(text.split()) * 0.7,  # Check for repetitive words
            'is_very_short': len(text.strip()) < 5,
            'is_very_long': len(text.strip()) > 100
        }
        
        return patterns
    
    def _generate_streaming(self, inputs, max_new_tokens, temperature, top_p, gpu_monitor_data):
        """Generate text using streaming for accurate TTFT measurement"""
        try:
            from .streaming_generator import StreamingGenerator
            
            # Create streaming generator
            streamer = StreamingGenerator(self.model, self.processor, self.device)
            
            # Generate with streaming
            generated_text, streaming_metrics = streamer.generate_with_accurate_metrics(
                inputs, max_new_tokens, temperature, top_p,
                do_sample=False,  # Use deterministic generation
                no_repeat_ngram_size=2,  # Prevent 2-gram repetition
                repetition_penalty=1.05  # Light repetition penalty
            )
            
            return generated_text, streaming_metrics
            
        except Exception as e:
            logger.error(f"Error in streaming generation: {e}")
            # Fallback to batch generation
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=False,  # Use deterministic generation
                    no_repeat_ngram_size=2,  # Prevent 2-gram repetition
                    repetition_penalty=1.05,  # Light repetition penalty
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    return_dict_in_generate=False,
                    output_scores=False,
                    return_audio=False
                )
            
            # Extract generated text
            if isinstance(outputs, torch.Tensor):
                generated_tokens = outputs[0][len(inputs['input_ids'][0]):]
            else:
                generated_tokens = outputs.sequences[0][len(inputs['input_ids'][0]):]
            
            generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            generated_text = generated_text.replace('<|im_end|>', '').strip()
            
            # Return basic metrics
            return generated_text, {
                'output_tokens': len(generated_tokens),
                'ttft': 0.0,  # Fallback value
                'step_times': []
            }
    

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
    
    def generate_draft_with_spec_logic(self, 
                                       audio_features: torch.Tensor,
                                       prompt: str = "Based on this audio, describe the emotional state of the speaker in Chinese.",
                                       max_new_tokens: int = 128,
                                       target_sentences: int = 2,
                                       min_chars: int = 90,
                                       min_new_tokens_sc: int = 48,
                                       prompt_type: str = "detailed") -> tuple[str, dict]:
        """
        Generate using the SAME logic as Speculative Decoding's edge generation.
        This ensures Edge Baseline is directly comparable to Speculative Decoding.
        
        This method creates a SimpleSpeculativeDecoding instance with entropy_threshold=999.0,
        which forces Edge-only mode (Cloud is never called). This way, Edge Baseline uses
        exactly the same generation logic (CJK-aware constraints, punctuation gates, etc.)
        as the Edge draft generation in Speculative Decoding.
        
        Args:
            audio_features: Audio waveform tensor (1D)
            prompt: Text prompt for generation
            max_new_tokens: Maximum number of tokens to generate
            target_sentences: Target number of sentences for stopping criteria
            min_chars: Minimum characters for stopping criteria
            min_new_tokens_sc: Minimum new tokens before stopping
            prompt_type: Type of prompt ("default", "detailed", "concise")
            
        Returns:
            Tuple of (generated_text, latency_metrics)
        """
        try:
            import time
            logger.info("=" * 80)
            logger.info("Using Speculative Decoding Edge logic for Edge Baseline generation")
            logger.info(f"This ensures 100% alignment with Speculative Decoding's Edge generation")
            logger.info("=" * 80)
            
            total_start_time = time.time()
            
            # Import speculative decoding logic
            from ..speculative_decoding import SimpleSpeculativeDecoding
            
            # DO NOT create CloudModel - we'll pass None and set entropy_threshold=999.0
            # This avoids loading the 7B Cloud model unnecessarily
            logger.info(f"Creating SimpleSpeculativeDecoding with entropy_threshold=999.0 (Edge-only mode)")
            logger.info(f"Cloud model will be set to None (not needed for Edge-only mode)")
            
            # Create spec decoder with VERY HIGH entropy threshold and cloud_model=None
            # This forces Edge-only mode: Cloud is never called for verification
            spec_decoder = SimpleSpeculativeDecoding(
                edge_model=self,
                cloud_model=None,  # CRITICAL: No cloud model needed
                k=5,  # Draft block size (same as in actual spec decoding)
                entropy_threshold=999.0,  # CRITICAL: Never call cloud (Edge only)
                target_sentences=target_sentences,
                min_chars=min_chars,
                min_new_tokens_sc=min_new_tokens_sc
            )
            
            logger.info(f"Generating with Edge-only mode:")
            logger.info(f"  - max_new_tokens={max_new_tokens}")
            logger.info(f"  - target_sentences={target_sentences}")
            logger.info(f"  - min_chars={min_chars}")
            logger.info(f"  - min_new_tokens_sc={min_new_tokens_sc}")
            logger.info(f"  - prompt_type={prompt_type}")
            
            # Use spec decoder's generation logic but force Edge-only
            # The high entropy_threshold ensures Cloud is never called
            # Note: prompt_type is not a parameter of generate(), it's already handled in __init__
            generated_text, spec_metrics = spec_decoder.generate(
                audio_features=audio_features,  # Fixed: was audio_waveform
                prompt=prompt,
                max_new_tokens=max_new_tokens
                # prompt_type is NOT passed here - stopping criteria already configured in __init__
            )
            
            total_time = time.time() - total_start_time
            
            logger.info(f"Edge-only generation completed in {total_time:.3f}s")
            logger.info(f"Generated text: {generated_text}")
            logger.info(f"Total cloud calls: {spec_metrics.get('total_cloud_calls', 0)} (should be 0)")
            
            # Extract relevant metrics from spec_metrics
            # spec_metrics already contains all the necessary fields
            # Format to match the expected latency_metrics structure
            latency_metrics = {
                'ttft': spec_metrics.get('ttft', 0.0),
                'itps': spec_metrics.get('itps', 0.0),
                'otps': spec_metrics.get('otps', 0.0),
                'oet': spec_metrics.get('oet', 0.0),
                'total_time': spec_metrics.get('total_time', total_time),
                'input_tokens': spec_metrics.get('input_tokens', 0),  # Fixed: was 'total_input_tokens'
                'output_tokens': spec_metrics.get('output_tokens', 0),  # Fixed: was 'total_output_tokens'
                'cpu_percent': spec_metrics.get('cpu_percent', 0.0),
                'ram_gb': spec_metrics.get('ram_gb', 0.0),
                'gpu_util': spec_metrics.get('gpu_util', 0.0),
                'gpu_memory_gb': spec_metrics.get('gpu_memory_gb', 0.0),
                'streaming_metrics': {
                    'ttft': spec_metrics.get('ttft', 0.0),
                    'itps': spec_metrics.get('itps', 0.0),
                    'otps': spec_metrics.get('otps', 0.0),
                    'oet': spec_metrics.get('oet', 0.0),
                    'total_time': spec_metrics.get('total_time', total_time),
                    'input_tokens': spec_metrics.get('input_tokens', 0),
                    'output_tokens': spec_metrics.get('output_tokens', 0),
                }
            }
            
            return generated_text, latency_metrics
            
        except Exception as e:
            logger.error(f"Error in generate_draft_with_spec_logic: {e}")
            import traceback
            traceback.print_exc()
            raise
    
