"""
Streaming generator for accurate TTFT measurement using TextIteratorStreamer
"""

import torch
import time
from typing import Iterator, Tuple, Dict, Any, List
import logging
from transformers import TextIteratorStreamer

logger = logging.getLogger(__name__)


class StreamingGenerator:
    """Streaming generator for accurate latency measurements using TextIteratorStreamer"""
    
    def __init__(self, model, processor, device: str = "cuda"):
        self.model = model
        self.processor = processor
        self.device = device
    
    def generate_streaming(self, 
                          inputs: Dict[str, torch.Tensor],
                          max_new_tokens: int = 32,
                          temperature: float = 0.7,
                          top_p: float = 0.9) -> Iterator[Tuple[str, float, Dict[str, Any]]]:
        """
        Generate text using TextIteratorStreamer for true streaming
        
        Args:
            inputs: Model inputs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Yields:
            Tuple of (text_chunk, generation_time, metrics)
        """
        try:
            import threading
            
            # Create TextIteratorStreamer
            streamer = TextIteratorStreamer(
                self.processor.tokenizer,
                timeout=60.0,
                skip_special_tokens=True,
                skip_prompt=True
            )
            
            # Prepare generation parameters
            generation_kwargs = {
                'max_new_tokens': max_new_tokens,
                'temperature': temperature,
                'top_p': top_p,
                'do_sample': True,
                'pad_token_id': self.processor.tokenizer.eos_token_id,
                'streamer': streamer,
                'return_dict_in_generate': False,
                'output_scores': False,
                'return_audio': False
            }
            
            # Track timing
            first_token_time = None
            total_start_time = time.time()
            token_count = 0
            generated_text = ""
            
            # Start generation in a separate thread
            generation_thread = threading.Thread(
                target=self.model.generate,
                kwargs={**inputs, **generation_kwargs}
            )
            generation_thread.start()
            
            # Stream tokens as they are generated
            for new_text in streamer:
                current_time = time.time()
                
                # Record first token time
                if first_token_time is None:
                    first_token_time = current_time
                    ttft = first_token_time - total_start_time
                else:
                    ttft = None
                
                # Calculate metrics
                step_time = current_time - total_start_time
                generated_text += new_text
                
                # Count tokens in the new text
                new_tokens = self.processor.tokenizer.encode(new_text, add_special_tokens=False)
                token_count += len(new_tokens)
                
                # Prepare metrics
                metrics = {
                    'step_time': step_time,
                    'total_time': current_time - total_start_time,
                    'ttft': ttft,
                    'token_count': token_count,
                    'generated_text': generated_text
                }
                
                # Yield text chunk and metrics
                yield new_text, step_time, metrics
                
                # Check for EOS tokens
                if any(eos_token in new_text for eos_token in ['<|im_end|>', '<|endoftext|>']):
                    break
            
            # Wait for generation to complete
            generation_thread.join()
            
        except Exception as e:
            logger.error(f"Error in streaming generation: {e}")
            raise
    
    def generate_with_accurate_metrics(self,
                                     inputs: Dict[str, torch.Tensor],
                                     max_new_tokens: int = 32,
                                     temperature: float = 0.7,
                                     top_p: float = 0.9) -> Tuple[str, Dict[str, Any]]:
        """
        Generate text with accurate latency metrics using TextIteratorStreamer
        
        Returns:
            Tuple of (generated_text, accurate_metrics)
        """
        try:
            total_start_time = time.time()
            
            # Collect all text chunks and timing data
            text_chunks = []
            step_times = []
            first_token_time = None
            ttft = None
            total_tokens = 0
            
            for text_chunk, step_time, metrics in self.generate_streaming(inputs, max_new_tokens, temperature, top_p):
                text_chunks.append(text_chunk)
                step_times.append(step_time)
                
                if first_token_time is None and metrics['ttft'] is not None:
                    first_token_time = metrics['ttft']
                    ttft = first_token_time
                
                total_tokens = metrics['token_count']
            
            total_end_time = time.time()
            
            # Calculate accurate metrics
            if not text_chunks:
                return "", {}
            
            # Combine all text chunks
            generated_text = ''.join(text_chunks)
            generated_text = generated_text.replace('<|im_end|>', '').strip()
            
            # Calculate metrics
            total_generation_time = total_end_time - total_start_time
            input_tokens = inputs['input_ids'].shape[1]
            output_tokens = total_tokens
            
            # Accurate TTFT (Time to First Token)
            accurate_ttft = ttft if ttft is not None else total_generation_time
            
            # Input Token Per Second
            input_processing_time = total_start_time - total_start_time  # This should be measured separately
            itps = input_tokens / input_processing_time if input_processing_time > 0 else 0
            
            # Output Token Per Second (excluding TTFT)
            generation_time = total_generation_time - accurate_ttft if accurate_ttft < total_generation_time else total_generation_time
            otps = output_tokens / generation_time if generation_time > 0 else 0
            
            # Output Evaluation Time
            oet = total_generation_time
            
            # Total Time
            total_time = total_end_time - total_start_time
            
            accurate_metrics = {
                'ttft': accurate_ttft,
                'itps': itps,
                'otps': otps,
                'oet': oet,
                'total_time': total_time,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'step_times': step_times,
                'first_token_time': first_token_time,
                'text_chunks': text_chunks
            }
            
            return generated_text, accurate_metrics
            
        except Exception as e:
            logger.error(f"Error in generate_with_accurate_metrics: {e}")
            return "", {}
