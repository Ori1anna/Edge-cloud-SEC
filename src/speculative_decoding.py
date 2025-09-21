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
                 prob_threshold: float = 0.25,
                 k: int = 5):
        """
        Initialize speculative decoding system
        
        Args:
            edge_model: Edge model for drafting
            cloud_model: Cloud model for verification
            entropy_threshold: Threshold for entropy-based uncertainty
            prob_threshold: Threshold for token acceptance
            k: Number of draft tokens to generate
        """
        self.edge_model = edge_model
        self.cloud_model = cloud_model
        self.entropy_threshold = entropy_threshold
        self.prob_threshold = prob_threshold
        self.k = k
        
        logger.info(f"Initialized SimpleSpeculativeDecoding with entropy_threshold={entropy_threshold}, "
                   f"prob_threshold={prob_threshold}, k={k}")
    
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
    
    def _update_context(self, context: dict, new_tokens: List[int]) -> dict:
        """Update context with new tokens"""
        try:
            if not new_tokens:
                return context
            
            # Convert to tensor
            new_tokens_tensor = torch.tensor([new_tokens], device=self.edge_model.device)
            
            # Update input_ids
            new_input_ids = torch.cat([context['input_ids'], new_tokens_tensor], dim=1)
            
            # Update attention_mask
            new_attention_mask = torch.cat([context['attention_mask'], 
                                          torch.ones((1, len(new_tokens)), device=self.edge_model.device)], dim=1)
            
            return {
                'input_ids': new_input_ids,
                'attention_mask': new_attention_mask
            }
            
        except Exception as e:
            logger.error(f"Error updating context: {e}")
            return context
    
    def _is_eos_token(self, token_id: int) -> bool:
        """Check if token is EOS token"""
        eos_token_id = self.edge_model.processor.tokenizer.eos_token_id
        im_end_id = self.edge_model.processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
        endoftext_id = self.edge_model.processor.tokenizer.convert_tokens_to_ids('<|endoftext|>')
        
        return token_id in [eos_token_id, im_end_id, endoftext_id]
    
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
            
            # Main speculative decoding loop
            while len(generated_tokens) < max_new_tokens:
                # Step 1: Edge model generates k draft tokens
                logger.info(f"Edge generating {self.k} draft tokens...")
                draft_start_time = time.time()
                draft_tokens, draft_logits = self._generate_draft_tokens(current_context, self.k)
                draft_end_time = time.time()
                
                # Record TTFT when first draft tokens are generated
                if first_token_time is None and draft_tokens:
                    first_token_time = draft_end_time
                
                if not draft_tokens:
                    logger.info("Edge model finished generation")
                    break
                
                total_draft_tokens += len(draft_tokens)
                logger.info(f"Edge generated {len(draft_tokens)} tokens: {draft_tokens}")
                
                # Step 2: Calculate uncertainty (entropy) for each draft token
                uncertainties = self._calculate_entropy_uncertainty(draft_logits)
                logger.info(f"Token uncertainties: {uncertainties}")
                
                # Step 3: Check if we need Cloud verification
                max_uncertainty = max(uncertainties) if uncertainties else 0
                needs_cloud_verification = max_uncertainty > self.entropy_threshold
                
                if needs_cloud_verification:
                    logger.info(f"High uncertainty ({max_uncertainty:.3f} > {self.entropy_threshold}), calling Cloud for verification")
                    cloud_calls += 1
                    
                    # Step 4: Cloud model verifies draft tokens
                    # 构造瘦身上下文，不传音频特征
                    ctx4cloud = {
                        "input_ids": current_context["input_ids"],
                        "attention_mask": current_context["attention_mask"],
                    }
                    accepted_tokens, correction_token, needs_correction = self.cloud_model.verify_tokens(
                        ctx4cloud, draft_tokens, self.prob_threshold
                    )
                    
                    if needs_correction:
                        total_corrections += 1
                        logger.info(f"Cloud correction: accepted {len(accepted_tokens)}/{len(draft_tokens)} tokens, correction: {correction_token}")
                        
                        # TTFT already recorded when first draft tokens were generated
                        
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
                        
                        # Step 5: Pass back to Edge
                        # Update context with accepted tokens + correction token (discard rejected tokens)
                        # 防止将 None 追加进上下文
                        append_list = accepted_tokens + ([correction_token] if correction_token is not None else [])
                        current_context = self._update_context(current_context, append_list)
                        
                        # Step 6: Continue generation from corrected position
                        logger.info("Continuing generation from corrected position...")
                    else:
                        logger.info(f"Cloud accepted all {len(draft_tokens)} tokens")
                        
                        # TTFT already recorded when first draft tokens were generated
                        
                        generated_tokens.extend(draft_tokens)
                        total_accepted_tokens += len(draft_tokens)
                        current_context = self._update_context(current_context, draft_tokens)
                else:
                    logger.info(f"Low uncertainty ({max_uncertainty:.3f} <= {self.entropy_threshold}), accepting all Edge tokens")
                    
                    # TTFT already recorded when first draft tokens were generated
                    
                    generated_tokens.extend(draft_tokens)
                    total_accepted_tokens += len(draft_tokens)
                    current_context = self._update_context(current_context, draft_tokens)
                
                # Check for EOS token
                if self.edge_model.processor.tokenizer.eos_token_id in generated_tokens:
                    logger.info("EOS token found, stopping generation")
                    break
            
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
            
            # Get CPU and GPU usage
            try:
                process = psutil.Process()
                cpu_percent = process.cpu_percent(interval=0.5)
                if cpu_percent == 0.0:
                    system_cpu = psutil.cpu_percent(interval=0.2)
                    cpu_percent = min(system_cpu * 0.1, 100.0)
                if cpu_percent == 0.0:
                    cpu_percent = 0.1
                
                memory_info = process.memory_info()
                ram_gb = memory_info.rss / (1024 ** 3)
                
                # GPU usage from monitoring
                gpu_util = 0.0
                gpu_memory_gb = 0.0
                if gpu_monitor_data and gpu_monitor_data.get('gpu_util_samples'):
                    gpu_util_samples = gpu_monitor_data['gpu_util_samples']
                    gpu_util = max(gpu_util_samples) if gpu_util_samples else 0.0
                    gpu_memory_gb = gpu_monitor_data.get('gpu_memory_gb', 0.0)
            except Exception as e:
                logger.warning(f"Failed to get CPU/GPU usage: {e}")
                cpu_percent = 0.0
                ram_gb = 0.0
                gpu_util = 0.0
                gpu_memory_gb = 0.0
            
            # Calculate latency metrics (compatible with baseline format)
            # Time-to-First-Token (TTFT): Time from generation start to first draft token generation
            # This measures the time until Edge model generates its first draft tokens
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
                'cpu_percent': cpu_percent,
                'ram_gb': ram_gb,
                'gpu_util': gpu_util,
                'gpu_memory_gb': gpu_memory_gb,
                
                # Speculative decoding specific metrics
                'cloud_calls': cloud_calls,
                'total_draft_tokens': total_draft_tokens,
                'total_accepted_tokens': total_accepted_tokens,
                'total_corrections': total_corrections,
                'acceptance_rate': total_accepted_tokens / total_draft_tokens if total_draft_tokens > 0 else 0,
                'correction_rate': total_corrections / cloud_calls if cloud_calls > 0 else 0,
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
    
    def _generate_draft_tokens(self, context: dict, k: int) -> tuple[list, torch.Tensor]:
        """
        Generate k draft tokens using Edge model
        
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
                        # Fallback if no scores available
                        vocab_size = self.edge_model.processor.tokenizer.vocab_size
                        logits = torch.zeros(k, vocab_size, device=self.edge_model.device)
                        logger.debug(f"Using fallback dummy logits shape: {logits.shape}")
                else:
                    # Fallback: create dummy logits
                    vocab_size = self.edge_model.processor.tokenizer.vocab_size
                    logits = torch.zeros(k, vocab_size, device=self.edge_model.device)
                    logger.debug(f"Using dummy logits shape: {logits.shape}")
                
                logger.debug(f"Generated {len(draft_tokens)} draft tokens: {draft_tokens}")
                return draft_tokens, logits
                
        except Exception as e:
            logger.error(f"Error in _generate_draft_tokens: {e}")
            import traceback
            traceback.print_exc()
            return [], torch.tensor([])
    
    def _calculate_entropy_uncertainty(self, logits: torch.Tensor) -> list[float]:
        """
        Calculate entropy uncertainty for each token position
        
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
            
            # Calculate probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Calculate entropy: H = -sum(p * log(p))
            # Add small epsilon to avoid log(0) which causes negative values
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            # Clamp negative values to 0 (they indicate numerical issues)
            entropy = torch.clamp(entropy, min=0.0)
            
            # Convert to list
            entropy_list = entropy.cpu().tolist()
            
            logger.debug(f"Calculated entropy uncertainties for {len(entropy_list)} tokens: {entropy_list}")
            return entropy_list
            
        except Exception as e:
            logger.error(f"Error calculating entropy uncertainty: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _update_context(self, context: dict, new_tokens: list) -> dict:
        """
        Update context with new tokens
        
        Args:
            context: Current context dictionary
            new_tokens: List of new token IDs to add
            
        Returns:
            Updated context dictionary
        """
        try:
            if not new_tokens:
                return context
            
            # Convert new tokens to tensor
            new_tokens_tensor = torch.tensor(new_tokens, device=self.edge_model.device).unsqueeze(0)
            
            # Update input_ids
            updated_input_ids = torch.cat([context['input_ids'], new_tokens_tensor], dim=1)
            
            # Update attention_mask
            new_attention_mask = torch.ones((1, len(new_tokens)), device=self.edge_model.device)
            updated_attention_mask = torch.cat([context['attention_mask'], new_attention_mask], dim=1)
            
            # Create updated context
            updated_context = context.copy()
            updated_context['input_ids'] = updated_input_ids
            updated_context['attention_mask'] = updated_attention_mask
            
            logger.debug(f"Updated context with {len(new_tokens)} new tokens")
            return updated_context
            
        except Exception as e:
            logger.error(f"Error updating context: {e}")
            return context
