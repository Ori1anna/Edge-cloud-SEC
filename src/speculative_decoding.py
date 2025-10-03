"""
Simplified Speculative Decoding Implementation
Based on streaming generation with entropy-based uncertainty
"""

import torch
import time
import logging
from typing import Dict, List, Optional, Tuple
from src.models.edge_model import EdgeModel
from src.models.cloud_model import CloudModel

logger = logging.getLogger(__name__)


class SimpleSpeculativeDecoding:
    """Simplified speculative decoding system"""
    
    def __init__(self, 
                 edge_model: EdgeModel, 
                 cloud_model: CloudModel,
                 entropy_threshold: float = 1.5,
                 k: int = 5):
        """
        Initialize speculative decoding system
        
        Args:
            edge_model: Edge model for drafting
            cloud_model: Cloud model for verification
            entropy_threshold: Threshold for entropy-based uncertainty
            k: Number of draft tokens to generate
        """
        self.edge_model = edge_model
        self.cloud_model = cloud_model
        self.entropy_threshold = entropy_threshold
        self.k = k
        
        logger.info(f"Initialized SimpleSpeculativeDecoding with entropy_threshold={entropy_threshold}, k={k}")
    
    def _prepare_initial_context(self, audio_features: torch.Tensor, prompt: str) -> dict:
        """Prepare initial context from audio and prompt"""
        try:
            # Create conversation with audio input (same format as Edge model)
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
            text = self.edge_model.processor.apply_chat_template(
                conversation, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Debug: log the complete prompt
            logger.info(f"Speculative Decoding prompt: {text}")
            
            # Process inputs with audio waveform - ensure correct shape
            # 如果audio_features是2D，转换为1D
            if audio_features.dim() == 2 and audio_features.shape[0] == 1:
                audio_features = audio_features.squeeze(0)
            
            inputs = self.edge_model.processor(
                text=text,
                audio=audio_features,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.edge_model.device) for k, v in inputs.items()}
            
            # Debug: check if audio is properly included
            logger.info(f"Input keys: {list(inputs.keys())}")
            if 'input_features' in inputs:
                logger.info(f"Audio features shape: {inputs['input_features'].shape}")
            elif 'audio_features' in inputs:
                logger.info(f"Audio features shape: {inputs['audio_features'].shape}")
            else:
                logger.warning("No audio features found in inputs!")
            
            return inputs
            
        except Exception as e:
            logger.error(f"Error preparing initial context: {e}")
            raise
    
    
    def _is_eos_token(self, token_id: int) -> bool:
        """Check if token is EOS token or should stop generation"""
        eos_token_id = self.edge_model.processor.tokenizer.eos_token_id
        im_end_id = self.edge_model.processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
        endoftext_id = self.edge_model.processor.tokenizer.convert_tokens_to_ids('<|endoftext|>')
        
        # Standard EOS tokens
        if token_id in [eos_token_id, im_end_id, endoftext_id]:
            return True
        
        # Chinese punctuation and newline as stop conditions
        try:
            # Convert token to text to check for stop conditions
            token_text = self.edge_model.processor.tokenizer.decode([token_id], skip_special_tokens=True)
            
            # Stop on Chinese period, newline, or "Human" (conversation end)
            if token_text in ['。', '\n', 'Human', 'Human:', 'Please', 'functionalities', '—', '**', '"', '"', '```', '```', '我/', '我/']:
                return True
                
            # Stop if token contains newline (for multi-token newlines)
            if '\n' in token_text:
                return True
                
            # Stop on truly problematic patterns that indicate generation issues
            # Only stop on patterns that are universally problematic across datasets
            if token_text in ['<|endoftext|>', '<|im_end|>', '<|end|>']:
                return True
                
                
        except Exception:
            # If decoding fails, just check standard EOS tokens
            pass
        
        return False
    
    def _decode_tokens(self, tokens: List[int]) -> str:
        """Decode tokens to text"""
        try:
            text = self.edge_model.processor.tokenizer.decode(tokens, skip_special_tokens=True)
            text = text.replace('<|im_end|>', '').strip()
            return text
        except Exception as e:
            logger.error(f"Error decoding tokens: {e}")
            return ""
    
    def generate(self, 
                 audio_features: torch.Tensor, 
                 prompt: str = "基于这个音频，用中文描述说话人的情感状态。",
                 max_new_tokens: int = 32) -> Tuple[str, Dict]:
        """
        Main generation function using speculative decoding
        
        Args:
            audio_features: Audio input tensor
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Tuple of (generated_text, latency_metrics)
        """
        try:
            import psutil
            
            # Record start time for total processing
            total_start_time = time.time()
            
            # Prepare initial context
            context = self._prepare_initial_context(audio_features, prompt)
            
            # Record time after input processing
            input_end_time = time.time()
            input_processing_time = input_end_time - total_start_time
            
            # Metrics tracking
            cloud_calls = 0
            total_draft_tokens = 0
            total_accepted_tokens = 0
            total_corrections = 0
            
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
                            for _ in range(100):  # Sample for up to 10 seconds
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
            
            logger.info(f"Starting speculative decoding with max_new_tokens={max_new_tokens}")
            
            # Record generation start time
            generation_start_time = time.time()
            
            # Initialize generation state
            generated_tokens = []
            current_context = context.copy()
            first_token_time = None  # Track TTFT
            
            # Add timeout mechanism to prevent infinite loops
            max_generation_time = 300.0  # 5 minutes maximum
            generation_start_time = time.time()
            
            # Start hardware monitoring (similar to Edge Only approach)
            monitor_data = {
                'cpu_samples': [], 
                'memory_samples': [],
                'gpu_util_samples': [],
                'gpu_memory_gb': 0.0
            }
            monitor_thread = None
            
            def monitor_hardware():
                """Monitor hardware usage during speculative decoding generation"""
                start_time = time.time()
                while time.time() - start_time < 600:  # Monitor for up to 10 minutes
                    try:
                        import psutil
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
                        
                        monitor_data['cpu_samples'].append(max(cpu_percent, 0.1))  # Ensure minimum 0.1% to avoid 0 values
                        monitor_data['memory_samples'].append(memory_gb)
                        
                        # Monitor GPU usage (for Cloud model usage)
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
                        
                    except Exception as e:
                        logger.debug(f"Hardware monitoring error: {e}")
                        # Continue monitoring even if one sample fails
                    
                    time.sleep(0.1)  # Sample every 100ms
            
            # Start hardware monitoring thread
            import threading
            monitor_thread = threading.Thread(target=monitor_hardware)
            monitor_thread.daemon = True
            monitor_thread.start()
            logger.debug("Hardware monitoring thread started")
            
            # Step 0: Perform multimodal prefill to get KV cache
            logger.info("Performing multimodal prefill for efficient incremental generation...")
            prefill_start_time = time.time()
            current_context = self._prefill_multimodal_context(current_context)
            prefill_time = time.time() - prefill_start_time
            logger.info(f"Multimodal prefill completed in {prefill_time:.3f}s")
            
            # Main speculative decoding loop with incremental generation
            while len(generated_tokens) < max_new_tokens:
                # Check for timeout
                elapsed_time = time.time() - generation_start_time
                if elapsed_time > max_generation_time:
                    logger.warning(f"Generation timeout after {elapsed_time:.2f}s, stopping generation")
                    break
                
                # Step 1: Generate k draft tokens using incremental generation
                logger.info(f"Edge generating {self.k} draft tokens incrementally...")
                draft_start_time = time.time()
                
                # Cache KV state before draft generation for potential rollback
                kv_before_draft = current_context.get('past_key_values')
                ids_before_draft = current_context['input_ids'].shape[1]
                logger.debug(f"Before draft generation: input_ids length={ids_before_draft}, KV cache present={kv_before_draft is not None}")
                
                # Use incremental generation with KV cache
                try:
                    if 'past_key_values' in current_context:
                        logger.info("Using incremental generation with KV cache")
                        draft_tokens, draft_logits = self._generate_draft_tokens_incremental(current_context, self.k)
                    else:
                        # Fallback to legacy method if no KV cache
                        logger.warning("No KV cache available, falling back to legacy generation")
                        draft_tokens, draft_logits = self._generate_draft_tokens(current_context, self.k)
                except ValueError as e:
                    # Handle case where model generation fails to return scores
                    logger.error(f"Draft generation failed due to scores issue: {e}")
                    logger.warning("Skipping this generation round due to model issue")
                    continue
                except Exception as e:
                    # Handle other generation errors
                    logger.error(f"Draft generation failed with error: {e}")
                    logger.warning("Skipping this generation round due to error")
                    continue
                
                draft_end_time = time.time()
                draft_generation_time = draft_end_time - draft_start_time
                
                # Check for abnormally long generation time
                if draft_generation_time > 30.0:  # 30 seconds threshold
                    logger.warning(f"Abnormally long draft generation time: {draft_generation_time:.2f}s")
                
                # Check for empty or invalid tokens
                if not draft_tokens or len(draft_tokens) == 0:
                    logger.warning("Edge model generated no tokens, stopping generation")
                    break
                
                # Note: TTFT will be recorded when first tokens are actually accepted
                # This could be either Edge tokens (if uncertainty is low) or Cloud-verified tokens
                
                if not draft_tokens:
                    logger.info("Edge model finished generation")
                    break
                
                total_draft_tokens += len(draft_tokens)
                logger.info(f"Edge generated {len(draft_tokens)} tokens: {draft_tokens}")
                
                # Step 2: Calculate uncertainty (entropy) for each draft token
                uncertainties = self._calculate_entropy_uncertainty(draft_logits)
                logger.info(f"Token uncertainties: {uncertainties}")
                
                # Check for abnormal uncertainty patterns
                if uncertainties and all(u == 0.0 for u in uncertainties):
                    logger.warning("All uncertainties are 0.0 - this indicates a model issue!")
                    logger.warning("Forcing cloud verification due to abnormal uncertainty pattern")
                    needs_cloud_verification = True
                    max_uncertainty = self.entropy_threshold + 0.1  # Force above threshold
                else:
                    # Step 3: Check if we need Cloud verification
                    max_uncertainty = max(uncertainties) if uncertainties else 0
                    needs_cloud_verification = max_uncertainty > self.entropy_threshold
                
                if needs_cloud_verification:
                    logger.info(f"High uncertainty ({max_uncertainty:.3f} > {self.entropy_threshold}), calling Cloud for verification")
                    cloud_calls += 1
                    
                    # Step 4: Cloud model verifies draft tokens
                    # Pass full context including audio features
                    # Note: prob_threshold is no longer needed as we use rank-based threshold internally
                    accepted_tokens, correction_token, needs_correction = self.cloud_model.verify_tokens(
                        current_context, draft_tokens, None  # prob_threshold no longer used
                    )
                    
                    if needs_correction:
                        total_corrections += 1
                        should_stop = False  # Initialize should_stop for correction path
                        
                        # Record TTFT when first tokens are actually accepted
                        if first_token_time is None and (accepted_tokens or correction_token is not None):
                            first_token_time = time.time()
                            logger.debug(f"TTFT recorded: first accepted tokens from Cloud verification")
                        
                        # Step 4: Replace + Discard logic
                        # - Accept all tokens before the first rejection
                        # - Replace the first rejected token with Cloud's correction
                        # - Discard all tokens after the first rejection
                        
                        # Add accepted tokens to final output
                        generated_tokens.extend(accepted_tokens)
                        total_accepted_tokens += len(accepted_tokens)
                        
                        # Add correction token to final output
                        if correction_token is not None:
                            generated_tokens.append(correction_token)
                            total_accepted_tokens += 1
                            
                            # Check for stop condition after correction
                            if self._is_eos_token(correction_token):
                                logger.info(f"Stop condition met in correction token {correction_token}, stopping generation")
                                should_stop = True
                        
                        # CRITICAL: Fix KV consistency after correction
                        # Step 1: Rollback both KV and input_ids to state before draft generation
                        current_context['past_key_values'] = kv_before_draft
                        
                        # Rollback input_ids to length before draft generation
                        if ids_before_draft < current_context['input_ids'].shape[1]:
                            current_context['input_ids'] = current_context['input_ids'][:, :ids_before_draft]
                            current_context['attention_mask'] = current_context['attention_mask'][:, :ids_before_draft]
                            logger.info(f"Rolled back input_ids to length {ids_before_draft}")
                        
                        # Step 2: Advance KV cache with only accepted tokens + correction token
                        tokens_to_advance = accepted_tokens + ([correction_token] if correction_token is not None else [])
                        if kv_before_draft is not None and tokens_to_advance:
                            logger.info(f"Advancing KV cache with {len(tokens_to_advance)} corrected tokens: {tokens_to_advance}")
                            updated_kv = self._advance_kv_cache(kv_before_draft, tokens_to_advance, self.edge_model.device)
                            current_context['past_key_values'] = updated_kv
                        
                        # Step 3: Update context with accepted tokens + correction token (discard rejected tokens)
                        append_list = accepted_tokens + ([correction_token] if correction_token is not None else [])
                        current_context = self._update_context_incremental(current_context, append_list)
                        
                        # Step 6: Continue generation from corrected position
                        logger.info("Continuing generation from corrected position...")
                    else:
                        logger.info(f"Cloud accepted all {len(draft_tokens)} tokens")
                        should_stop = False  # Initialize should_stop for acceptance path
                        
                        # Record TTFT when first tokens are actually accepted
                        if first_token_time is None:
                            first_token_time = time.time()
                            logger.debug(f"TTFT recorded: all Edge tokens accepted by Cloud")
                        
                        generated_tokens.extend(draft_tokens)
                        total_accepted_tokens += len(draft_tokens)
                        current_context = self._update_context_incremental(current_context, draft_tokens)
                        
                        # Check for stop condition in accepted tokens
                        for token in draft_tokens:
                            if self._is_eos_token(token):
                                logger.info(f"Stop condition met in accepted token {token}, stopping generation")
                                should_stop = True
                                break
                else:
                    logger.info(f"Low uncertainty ({max_uncertainty:.3f} <= {self.entropy_threshold}), accepting all Edge tokens")
                    should_stop = False  # Initialize should_stop for low uncertainty path
                    
                    # Record TTFT when first tokens are actually accepted
                    if first_token_time is None:
                        first_token_time = time.time()
                        logger.debug(f"TTFT recorded: all Edge tokens accepted (low uncertainty)")
                    
                    generated_tokens.extend(draft_tokens)
                    total_accepted_tokens += len(draft_tokens)
                    current_context = self._update_context_incremental(current_context, draft_tokens)
                    
                    # Check for stop condition in accepted tokens
                    for token in draft_tokens:
                        if self._is_eos_token(token):
                            logger.info(f"Stop condition met in accepted token {token}, stopping generation")
                            should_stop = True
                            break
                
                # Check for EOS token or other stop conditions (if not already set)
                if not should_stop:
                    for token in generated_tokens[-5:]:  # Check last 5 tokens
                        if self._is_eos_token(token):
                            logger.info(f"Stop condition met (token {token}), stopping generation")
                            should_stop = True
                            break
                
                if should_stop:
                    break
                    
                # Minimal quality control: only stop on truly universal issues
                if len(generated_tokens) > 15:  # After generating some tokens
                    try:
                        current_text = self._decode_tokens(generated_tokens)
                        
                        # Stop only on model-specific stop tokens (universal across all models)
                        if any(pattern in current_text for pattern in ['<|endoftext|>', '<|im_end|>', '<|end|>']):
                            logger.info(f"Stopping due to model-specific stop tokens: {current_text}")
                            break
                            
                        # Stop if text is extremely long (prevent runaway generation)
                        if len(current_text) > 1000:  # Very generous limit for any dataset
                            logger.info(f"Stopping due to excessive text length: {len(current_text)} characters")
                            break
                            
                        # Stop if text is extremely repetitive (same token repeated many times)
                        if len(set(generated_tokens[-25:])) < 2 and len(generated_tokens) > 25:  # Very strict repetition check
                            logger.info(f"Stopping due to extreme repetition: {generated_tokens[-25:]}")
                            break
                            
                    except Exception as e:
                        logger.debug(f"Error checking generated text quality: {e}")
                        # Continue generation if text checking fails
            
            # Wait for GPU monitoring to complete
            if gpu_monitor_thread and gpu_monitor_thread.is_alive():
                gpu_monitor_thread.join(timeout=0.5)
            
            # Record final times
            generation_end_time = time.time()
            total_end_time = time.time()
            
            # Decode generated text
            generated_text = self._decode_tokens(generated_tokens)
            
            # Calculate input tokens (from initial context)
            input_tokens = len(context['input_ids'][0])
            output_tokens = len(generated_tokens)
            
            # Get CPU and GPU usage from monitoring data (similar to Edge Only approach)
            try:
                # Calculate average CPU and memory usage from monitoring samples
                if monitor_data['cpu_samples']:
                    cpu_percent = sum(monitor_data['cpu_samples']) / len(monitor_data['cpu_samples'])
                    ram_gb = sum(monitor_data['memory_samples']) / len(monitor_data['memory_samples'])
                    
                    # Calculate max values for additional metrics
                    cpu_max = max(monitor_data['cpu_samples'])
                    memory_max_gb = max(monitor_data['memory_samples'])
                else:
                    # Fallback if no monitoring data available
                    cpu_percent = 0.1
                    ram_gb = 0.0
                    cpu_max = 0.1
                    memory_max_gb = 0.0
                
                # GPU usage from monitoring
                gpu_util = 0.0
                gpu_memory_gb = 0.0
                if monitor_data['gpu_util_samples']:
                    gpu_util = max(monitor_data['gpu_util_samples'])
                    gpu_memory_gb = monitor_data['gpu_memory_gb']
                    
                logger.debug(f"Hardware monitoring completed: CPU avg={cpu_percent:.1f}%, RAM avg={ram_gb:.2f}GB, GPU max={gpu_util:.1f}%")
                    
            except Exception as e:
                logger.warning(f"Failed to process hardware monitoring data: {e}")
                cpu_percent = 0.1
                ram_gb = 0.0
                gpu_util = 0.0
                gpu_memory_gb = 0.0
                cpu_max = 0.1
                memory_max_gb = 0.0
            
            # Calculate latency metrics (compatible with baseline format)
            # Time-to-First-Token (TTFT): Time from generation start to first token acceptance
            # This measures the time until the first token is accepted (either by Cloud verification or low uncertainty)
            # For speculative decoding, TTFT represents when the first token is actually accepted and committed to output
            if first_token_time is not None:
                ttft = first_token_time - generation_start_time
            else:
                # Fallback if no tokens were generated
                ttft = generation_end_time - generation_start_time
            
            # Input Token Per Second (ITPS): Input processing speed
            itps = input_tokens / input_processing_time if input_processing_time > 0 else 0
            
            # Output Token Per Second (OTPS): Generation speed
            generation_time = generation_end_time - generation_start_time
            otps = output_tokens / generation_time if generation_time > 0 else 0
            
            # Output Evaluation Time (OET): Total generation time
            oet = generation_time
            
            # Total Time: Complete response generation time
            total_time = total_end_time - total_start_time
            
            latency_metrics = {
                # Standard latency metrics
                'ttft': ttft,
                'itps': itps,
                'otps': otps,
                'oet': oet,
                'total_time': total_time,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                
                # Hardware monitoring metrics (similar to Edge Only)
                'cpu_percent': cpu_percent,
                'cpu_avg': cpu_percent,  # Average CPU usage
                'cpu_max': cpu_max,      # Maximum CPU usage
                'ram_gb': ram_gb,
                'memory_avg_gb': ram_gb,  # Average memory usage
                'memory_max_gb': memory_max_gb,  # Maximum memory usage
                'gpu_util': gpu_util,
                'gpu_memory_gb': gpu_memory_gb,
                
                # Speculative decoding specific metrics
                'cloud_calls': cloud_calls,
                'total_draft_tokens': total_draft_tokens,
                'total_accepted_tokens': total_accepted_tokens,
                'total_corrections': total_corrections,
                'acceptance_rate': total_accepted_tokens / total_draft_tokens if total_draft_tokens > 0 else 0,
                'correction_rate': total_corrections / cloud_calls if cloud_calls > 0 else 0,
                'prefill_time': current_context.get('prefill_time', 0.0),
                'cloud_call_rate': cloud_calls / (total_draft_tokens // self.k + 1) if total_draft_tokens > 0 else 0
            }
            
            logger.info(f"Speculative decoding completed: {len(generated_tokens)} tokens, "
                       f"{cloud_calls} cloud calls, {total_corrections} corrections")
            
            return generated_text, latency_metrics
            
        except Exception as e:
            logger.error(f"Error in speculative decoding: {e}")
            import traceback
            traceback.print_exc()
            return "", {}
    
    def _prefill_multimodal_context(self, context: dict) -> dict:
        """
        Perform multimodal prefill to get KV cache for efficient incremental generation
        
        Args:
            context: Input context dictionary
            
        Returns:
            Enhanced context with KV cache
        """
        try:
            import time
            
            logger.info("Performing multimodal prefill with Edge model...")
            prefill_start_time = time.time()
            
            with torch.no_grad():
                # Prepare inputs for prefill using thinker
                inputs = {
                    'input_ids': context['input_ids'],
                    'attention_mask': context['attention_mask'],
                    'use_cache': True  # Enable KV caching
                }
                
                # Add audio features if present
                if 'input_features' in context:
                    inputs['input_features'] = context['input_features']
                if 'feature_attention_mask' in context:
                    inputs['feature_attention_mask'] = context['feature_attention_mask']
                
                # Use thinker for multimodal prefill (more efficient than full model)
                outputs = self.edge_model.model.thinker(**inputs, return_dict=True)
                
                # Extract past_key_values for incremental generation
                past_key_values = outputs.past_key_values
                
                # Debug: check outputs
                logger.debug(f"Prefill outputs logits shape: {outputs.logits.shape}")
                logger.debug(f"Prefill past_key_values type: {type(past_key_values)}")
                if past_key_values is not None:
                    logger.debug(f"Past_key_values length: {len(past_key_values)}")
                
                # Store the last token position for incremental generation
                last_token_id = context['input_ids'][0, -1:].clone()
                
                prefill_time = time.time() - prefill_start_time
                logger.info(f"Edge prefill completed in {prefill_time:.3f}s")
                
                # Return enhanced context with KV cache
                enhanced_context = context.copy()
                enhanced_context['past_key_values'] = past_key_values
                enhanced_context['last_token_id'] = last_token_id
                enhanced_context['prefill_time'] = prefill_time
                
                return enhanced_context
                
        except Exception as e:
            logger.error(f"Error in _prefill_multimodal_context: {e}")
            import traceback
            traceback.print_exc()
            # Return original context if prefill fails
            return context

    def _generate_draft_tokens_incremental(self, context: dict, k: int) -> tuple[list, torch.Tensor]:
        """
        Generate k draft tokens using incremental generation with KV cache
        
        Args:
            context: Context with past_key_values from prefill
            k: Number of tokens to generate
            
        Returns:
            Tuple of (draft_tokens, logits)
        """
        try:
            import time
            
            logger.debug(f"Generating {k} tokens incrementally with KV cache...")
            
            with torch.no_grad():
                draft_tokens = []
                all_logits = []
                current_past_key_values = context.get('past_key_values')
                
                # Start from the last token of the input sequence
                # For incremental generation, we only need the last token
                current_input_ids = context['input_ids'][0, -1:].unsqueeze(0)  # Shape: [1, 1]
                
                logger.debug(f"Starting incremental generation from token: {current_input_ids.item()}")
                
                # Generate k tokens incrementally
                for step in range(k):
                    # Prepare input for this step
                    step_inputs = {
                        'input_ids': current_input_ids,
                        'past_key_values': current_past_key_values,
                        'use_cache': True,
                        'return_dict': True
                    }
                    
                    # CRITICAL: Pass audio features on the first step to maintain multimodal context
                    if step == 0:
                        if 'input_features' in context:
                            step_inputs['input_features'] = context['input_features']
                        if 'feature_attention_mask' in context:
                            step_inputs['feature_attention_mask'] = context['feature_attention_mask']
                        logger.debug(f"Step {step+1}: Including audio features for multimodal context")
                    else:
                        logger.debug(f"Step {step+1}: Using KV cache only (audio context preserved)")
                    
                    # Debug: check input shapes
                    logger.debug(f"Step {step+1}: input_ids={current_input_ids.item()}, past_key_values present: {current_past_key_values is not None}")
                    
                    # Call thinker for incremental generation
                    outputs = self.edge_model.model.thinker(**step_inputs)
                    
                    # Get logits for this step - use float32 for better numerical stability
                    logits = outputs.logits[0, -1, :].float()  # Shape: [vocab_size], convert to float32
                    all_logits.append(logits)
                    
                    # Sample next token with enhanced anti-repetition logic
                    # Apply temperature and repetition penalty
                    logits_temp = logits / 0.7  # Temperature scaling
                    
                    # Enhanced repetition penalty: reduce probability of recently generated tokens
                    if len(draft_tokens) > 0:
                        recent_tokens = draft_tokens[-5:]  # Look at last 5 tokens
                        for token_id in recent_tokens:
                            if logits_temp[token_id] > 0:
                                logits_temp[token_id] = logits_temp[token_id] / 1.5  # Stronger penalty
                    
                    # Additional penalty for common problematic tokens
                    problematic_tokens = [
                        self.edge_model.processor.tokenizer.convert_tokens_to_ids('请'),
                        self.edge_model.processor.tokenizer.convert_tokens_to_ids('好'),
                        self.edge_model.processor.tokenizer.convert_tokens_to_ids('的'),
                        self.edge_model.processor.tokenizer.convert_tokens_to_ids('调'),
                        self.edge_model.processor.tokenizer.convert_tokens_to_ids('语'),
                        self.edge_model.processor.tokenizer.convert_tokens_to_ids('声'),
                        self.edge_model.processor.tokenizer.convert_tokens_to_ids('，'),
                    ]
                    
                    for token_id in problematic_tokens:
                        if token_id is not None and token_id < logits_temp.size(0) and logits_temp[token_id] > 0:
                            logits_temp[token_id] = logits_temp[token_id] / 2.0  # Heavy penalty for problematic tokens
                    
                    # Apply top-p filtering
                    sorted_logits, sorted_indices = torch.sort(logits_temp, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > 0.9  # top_p=0.9
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits_temp[indices_to_remove] = float('-inf')
                    
                    # Sample from filtered distribution
                    probs = torch.softmax(logits_temp, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
                    draft_tokens.append(next_token)
                    
                    # Update for next iteration - keep the KV cache!
                    current_past_key_values = outputs.past_key_values
                    current_input_ids = torch.tensor([[next_token]], device=self.edge_model.device)
                    
                    logger.debug(f"Incremental step {step+1}: generated token {next_token}")
                
                # Stack logits: [k, vocab_size]
                logits_tensor = torch.stack(all_logits, dim=0)
                
                # Update context with new KV cache for next iteration
                context['past_key_values'] = current_past_key_values
                context['last_generated_tokens'] = draft_tokens
                
                logger.debug(f"Generated {len(draft_tokens)} draft tokens incrementally: {draft_tokens}")
                logger.debug(f"Updated context with new KV cache for next iteration")
                return draft_tokens, logits_tensor
                
        except Exception as e:
            logger.error(f"Error in _generate_draft_tokens_incremental: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to legacy method
            logger.warning("Falling back to legacy draft generation method")
            return self._generate_draft_tokens(context, k)

    def _advance_kv_cache(self, kv_cache, tokens: list, device) -> tuple:
        """
        Advance KV cache with new tokens using incremental generation
        
        Args:
            kv_cache: Current KV cache (past_key_values)
            tokens: List of new tokens to advance with
            device: Device for tensor operations
            
        Returns:
            Updated KV cache
        """
        try:
            if not tokens:
                return kv_cache
            
            current_kv = kv_cache
            
            with torch.no_grad():
                for i, token in enumerate(tokens):
                    # Prepare input for this step - each token as a single element
                    current_input_ids = torch.tensor([[token]], device=device)
                    
                    step_inputs = {
                        'input_ids': current_input_ids,
                        'past_key_values': current_kv,
                        'use_cache': True,
                        'return_dict': True
                    }
                    
                    # Advance KV cache one token at a time
                    outputs = self.edge_model.model.thinker(**step_inputs)
                    current_kv = outputs.past_key_values
                    
                    logger.debug(f"KV advance step {i+1}: token {token}")
            
            logger.info(f"Successfully advanced KV cache with {len(tokens)} tokens")
            return current_kv
            
        except Exception as e:
            logger.error(f"Error advancing KV cache: {e}")
            import traceback
            traceback.print_exc()
            return kv_cache

    def _update_context_incremental(self, context: dict, new_tokens: list) -> dict:
        """
        Update context with new tokens for incremental generation
        
        Args:
            context: Current context with KV cache
            new_tokens: List of new tokens to add
            
        Returns:
            Updated context
        """
        try:
            if not new_tokens:
                return context
            
            # CRITICAL: Create a copy to preserve all multimodal features
            new_context = context.copy()  # Preserve input_features, feature_attention_mask, past_key_values, etc.
            
            # Update input_ids and attention_mask
            new_tokens_tensor = torch.tensor([new_tokens], device=context['input_ids'].device)
            new_context['input_ids'] = torch.cat([new_context['input_ids'], new_tokens_tensor], dim=1)
            new_context['attention_mask'] = torch.cat([
                new_context['attention_mask'], 
                torch.ones_like(new_tokens_tensor)
            ], dim=1)
            
            # Update last_token_id for next incremental generation
            new_context['last_token_id'] = new_tokens_tensor[0, -1:].unsqueeze(0)
            
            # Note: KV cache management is handled separately in draft generation and correction logic
            logger.debug(f"Updated context with {len(new_tokens)} new tokens")
            
            return new_context
            
        except Exception as e:
            logger.error(f"Error in _update_context_incremental: {e}")
            return context

    def _generate_draft_tokens(self, context: dict, k: int) -> tuple[list, torch.Tensor]:
        """
        Generate k draft tokens using Edge model (legacy method for compatibility)
        
        Args:
            context: Current context dictionary
            k: Number of tokens to generate
            
        Returns:
            Tuple of (draft_tokens, logits)
        """
        try:
            with torch.no_grad():
                # Use Edge model to generate k tokens
                generation_kwargs = {
                    'input_ids': context['input_ids'],
                    'attention_mask': context['attention_mask'],
                    'max_new_tokens': k,
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'do_sample': True,
                    'pad_token_id': self.edge_model.processor.tokenizer.eos_token_id,
                    'return_dict_in_generate': True,
                    'output_scores': True,
                    'return_audio': False
                }
                
                # Include audio features if present
                if 'input_features' in context:
                    generation_kwargs['input_features'] = context['input_features']
                if 'feature_attention_mask' in context:
                    generation_kwargs['feature_attention_mask'] = context['feature_attention_mask']
                
                outputs = self.edge_model.model.generate(**generation_kwargs)
                
                # Extract generated tokens (only take the first k tokens)
                generated_tokens = outputs.sequences[0][len(context['input_ids'][0]):]
                draft_tokens = generated_tokens[:k].cpu().tolist()  # Only take k tokens
                
                # Extract logits - only take the first k steps corresponding to draft tokens
                if hasattr(outputs, 'scores') and outputs.scores:
                    # Only take the first k scores corresponding to our draft tokens
                    # outputs.scores is a tuple of tensors, each of shape [batch_size, vocab_size]
                    # We only need the first k steps since we only generated k draft tokens
                    draft_scores = outputs.scores[:k]  # Take only first k steps
                    if draft_scores:
                        logits = torch.stack(draft_scores, dim=1)  # Shape: [batch_size, k, vocab_size]
                        logits = logits[0]  # Remove batch dimension: [k, vocab_size]
                        logger.debug(f"Extracted logits shape: {logits.shape}, expected: [{k}, vocab_size]")
                    else:
                        # No scores available in draft_scores - this is a serious issue
                        logger.error(f"No scores available in outputs.scores[:k] - this indicates a model generation issue")
                        raise ValueError("Model generation failed to return scores - cannot compute uncertainty")
                else:
                    # No scores at all - this is a serious issue
                    logger.error(f"Model generation returned no scores (outputs.scores is None/empty) - this indicates a model configuration issue")
                    raise ValueError("Model generation failed to return scores - cannot compute uncertainty")
                
                logger.debug(f"Generated {len(draft_tokens)} draft tokens: {draft_tokens}")
                return draft_tokens, logits
                
        except Exception as e:
            logger.error(f"Error in _generate_draft_tokens: {e}")
            import traceback
            traceback.print_exc()
            return [], torch.tensor([])
    
    def _calculate_entropy_uncertainty(self, logits: torch.Tensor) -> list[float]:
        """
        Calculate entropy uncertainty for each token position using stable computation
        
        Args:
            logits: Logits tensor of shape [k, vocab_size]
            
        Returns:
            List of entropy values for each token position
        """
        try:
            if logits.numel() == 0:
                logger.warning("Empty logits tensor provided")
                return []
            
            logger.debug(f"Input logits shape: {logits.shape}")
            
            # Convert to float32 for better numerical stability
            logits_f32 = logits.float()
            
            # Debug: check if logits are all zeros
            if torch.all(logits_f32 == 0):
                logger.warning("All logits are zero! This indicates a model issue.")
                logger.debug(f"Logits stats: min={logits_f32.min()}, max={logits_f32.max()}, mean={logits_f32.mean()}")
            
            # Use log_softmax for numerical stability: log(p) = log_softmax(logits)
            log_probs = torch.log_softmax(logits_f32, dim=-1)
            probs = torch.exp(log_probs)  # Get probabilities from log probabilities
            
            # Calculate entropy: H = -sum(p * log(p))
            # Since we have log(p), we can compute -sum(p * log(p)) more stably
            entropy = -torch.sum(probs * log_probs, dim=-1)
            
            # Clamp negative values to 0 (they indicate numerical issues)
            entropy = torch.clamp(entropy, min=0.0)
            
            # Debug: check entropy calculation
            logger.debug(f"Logits stats: min={logits_f32.min()}, max={logits_f32.max()}, mean={logits_f32.mean()}")
            logger.debug(f"Entropy stats: min={entropy.min()}, max={entropy.max()}, mean={entropy.mean()}")
            
            # Check for extremely high entropy values that might indicate issues
            max_entropy = entropy.max().item()
            if max_entropy > 10.0:  # Log entropy > 10 is suspicious
                logger.warning(f"Extremely high entropy detected: {max_entropy:.2f}")
                logger.warning("This might indicate numerical issues or model instability")
            
            # Convert to list
            entropy_list = entropy.cpu().tolist()
            
            logger.debug(f"Calculated entropy uncertainties for {len(entropy_list)} tokens: {entropy_list}")
            return entropy_list
            
        except Exception as e:
            logger.error(f"Error calculating entropy uncertainty: {e}")
            import traceback
            traceback.print_exc()
            return []
