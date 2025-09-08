"""
Main speculative decoder controller
"""

import torch
from typing import Dict, List, Optional, Tuple
import logging
from .edge_model import EdgeModel
from .cloud_model import CloudModel

logger = logging.getLogger(__name__)


class SpeculativeDecoder:
    """Main controller for edge-cloud speculative decoding"""
    
    def __init__(self, 
                 edge_model: EdgeModel,
                 cloud_model: CloudModel,
                 config: Dict):
        self.edge_model = edge_model
        self.cloud_model = cloud_model
        self.config = config
        
        # Uncertainty thresholds
        self.entropy_threshold = config.get('entropy_threshold', 2.0)
        self.margin_threshold = config.get('margin_threshold', 0.1)
        self.log_prob_threshold = config.get('log_prob_threshold', -2.0)
        
        # Content-aware rules
        self.content_rules = config.get('content_rules', {})
        
    def decode(self, 
               audio_features: torch.Tensor,
               max_tokens: int = 100,
               temperature: float = 0.7) -> List[int]:
        """
        Main decoding function with edge-cloud speculation
        
        Args:
            audio_features: Audio features tensor
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated token sequence
        """
        generated_tokens = []
        
        while len(generated_tokens) < max_tokens:
            # Step 1: Generate draft on edge device
            draft_tokens, log_probs = self.edge_model.generate_draft(
                audio_features, max_new_tokens=32, temperature=temperature
            )
            
            # Step 2: Compute uncertainty signals
            uncertainty_signals = self.edge_model.compute_uncertainty(
                draft_tokens, log_probs
            )
            
            # Step 3: Decide whether to verify with cloud
            should_verify = self._should_verify(draft_tokens, uncertainty_signals)
            
            if should_verify:
                # Step 4: Verify with cloud model
                accepted_tokens, verification_length = self.cloud_model.verify_tokens(
                    draft_tokens, audio_features, generated_tokens
                )
                
                # Step 5: Update generated tokens
                generated_tokens.extend(accepted_tokens)
                
                # Step 6: Handle rollback if needed
                if verification_length < len(draft_tokens):
                    # Rollback and regenerate
                    rollback_tokens = self.cloud_model.rollback_and_regenerate(
                        generated_tokens, audio_features
                    )
                    generated_tokens.extend(rollback_tokens)
            else:
                # Accept all draft tokens
                generated_tokens.extend(draft_tokens)
            
            # Check for end token
            if self.edge_model.tokenizer.eos_token_id in generated_tokens:
                break
                
        return generated_tokens
    
    def _should_verify(self, 
                      tokens: List[int], 
                      uncertainty_signals: Dict[str, float]) -> bool:
        """
        Determine if tokens should be verified with cloud model
        
        Args:
            tokens: Draft tokens
            uncertainty_signals: Uncertainty measures
            
        Returns:
            True if verification is needed
        """
        # Check uncertainty thresholds
        if uncertainty_signals.get('entropy', 0) > self.entropy_threshold:
            return True
            
        if uncertainty_signals.get('margin', 0) < self.margin_threshold:
            return True
            
        if min(uncertainty_signals.get('token_log_probs', [0])) < self.log_prob_threshold:
            return True
        
        # Check content-aware rules
        if self._check_content_rules(tokens):
            return True
            
        return False
    
    def _check_content_rules(self, tokens: List[int]) -> bool:
        """
        Check content-aware verification rules
        
        Args:
            tokens: Tokens to check
            
        Returns:
            True if content rules require verification
        """
        # Convert tokens to text for rule checking
        text = self.edge_model.tokenizer.decode(tokens)
        
        # Check for numbers
        if self.content_rules.get('verify_numbers', True):
            if any(char.isdigit() for char in text):
                return True
        
        # Check for dates
        if self.content_rules.get('verify_dates', True):
            # Simple date pattern check
            import re
            date_patterns = [
                r'\d{1,2}/\d{1,2}/\d{2,4}',
                r'\d{4}-\d{2}-\d{2}',
                r'\d{1,2}-\d{1,2}-\d{2,4}'
            ]
            for pattern in date_patterns:
                if re.search(pattern, text):
                    return True
        
        # Check for names (capitalized words)
        if self.content_rules.get('verify_names', True):
            words = text.split()
            if any(word[0].isupper() and len(word) > 1 for word in words):
                return True
                
        return False
