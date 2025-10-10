"""
Proper stopping criteria for speculative decoding
Based on token IDs rather than string matching
"""

import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from typing import List


class StopWhenNSentences(StoppingCriteria):
    """
    Stop generation after N complete sentences (more flexible for detailed outputs)
    Uses token IDs for reliable stopping, not string matching
    """
    
    def __init__(self, tokenizer, n_sentences=2, sentence_end_chars=("。", "."), min_new_tokens=32):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            n_sentences: Number of sentences to allow before stopping
            sentence_end_chars: Characters that indicate end of sentence
            min_new_tokens: Minimum tokens to generate before allowing stop
        """
        self.n_sentences = n_sentences
        self.min_new_tokens = min_new_tokens
        self.generated_count = 0
        self.sentence_count = 0
        
        # Get token IDs for sentence-ending characters
        self.sentence_end_ids = []
        for char in sentence_end_chars:
            tokens = tokenizer.encode(char, add_special_tokens=False)
            if tokens:
                self.sentence_end_ids.append(tokens[-1])  # Use the last token ID
        
        # Also include EOS token IDs (these always stop generation)
        self.eos_ids = []
        if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
            self.eos_ids.append(tokenizer.eos_token_id)
        
        # Include common stop sequence IDs if available
        for stop_seq in ['<|im_end|>', '<|endoftext|>', '<|end|>']:
            if hasattr(tokenizer, 'convert_tokens_to_ids'):
                token_id = tokenizer.convert_tokens_to_ids(stop_seq)
                if token_id != tokenizer.unk_token_id:
                    self.eos_ids.append(token_id)
        
        # Remove duplicates while preserving order
        self.sentence_end_ids = list(dict.fromkeys(self.sentence_end_ids))
        self.eos_ids = list(dict.fromkeys(self.eos_ids))
        
        # Combine all stop IDs for EOS detection
        self.all_stop_ids = list(dict.fromkeys(self.sentence_end_ids + self.eos_ids))
    
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> bool:
        """
        Check if generation should stop
        
        Args:
            input_ids: Current input token IDs [batch_size, seq_len]
            scores: Current logits [batch_size, vocab_size]
            
        Returns:
            True if generation should stop, False otherwise
        """
        # Count newly generated tokens (excluding the original input)
        self.generated_count += 1
        
        # Check if the last generated token is a stop token
        last_token_id = input_ids[0, -1].item()
        
        # Always stop on EOS tokens (model-level stopping)
        if last_token_id in self.eos_ids:
            return True
        
        # For sentence-ending characters, count sentences
        if last_token_id in self.sentence_end_ids:
            self.sentence_count += 1
            
            # Stop if we've reached the target number of sentences AND minimum tokens
            if (self.sentence_count >= self.n_sentences and 
                self.generated_count >= self.min_new_tokens):
                return True
            
        return False


class StopAtFirstSentence(StopWhenNSentences):
    """
    Backward compatibility: Stop at first sentence (legacy behavior)
    """
    def __init__(self, tokenizer, stop_chars=("。", ".", "！", "!", "？", "?"), min_new_tokens=24):
        super().__init__(tokenizer, n_sentences=1, sentence_end_chars=stop_chars, min_new_tokens=min_new_tokens)


def create_stopping_criteria(tokenizer, 
                           n_sentences=2, 
                           sentence_end_chars=("。", "."), 
                           min_new_tokens=32,
                           prompt_type="default"):
    """
    Create stopping criteria for generation with flexible sentence counting
    
    Args:
        tokenizer: HuggingFace tokenizer
        n_sentences: Number of sentences to allow before stopping
        sentence_end_chars: Characters that indicate end of sentence
        min_new_tokens: Minimum tokens to generate before allowing stop
        prompt_type: Type of prompt (affects default parameters)
        
    Returns:
        StoppingCriteriaList with proper stopping criteria
    """
    # Adjust parameters based on prompt type
    if prompt_type == "detailed":
        # For detailed prompts, allow more sentences and tokens
        n_sentences = max(n_sentences, 2)  # At least 2 sentences
        min_new_tokens = max(min_new_tokens, 32)  # At least 32 tokens
    elif prompt_type == "concise":
        # For concise prompts, be more restrictive
        n_sentences = min(n_sentences, 1)  # At most 1 sentence
        min_new_tokens = min(min_new_tokens, 24)  # At most 24 tokens
    # For default prompts, use provided parameters
    
    stop_criteria = StopWhenNSentences(
        tokenizer=tokenizer,
        n_sentences=n_sentences,
        sentence_end_chars=sentence_end_chars,
        min_new_tokens=min_new_tokens
    )
    
    return StoppingCriteriaList([stop_criteria])
