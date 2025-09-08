#!/usr/bin/env python3
"""
Debug script to check uncertainty values in blocks
"""

import sys
import os
import torch
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.edge_model import EdgeModel
from models.cloud_model import CloudModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def debug_uncertainty_values():
    """Debug uncertainty values in generated blocks"""
    logger.info("Debugging uncertainty values...")
    
    # Initialize edge model
    try:
        edge_model = EdgeModel(device="cuda")
        logger.info("Edge model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load edge model: {e}")
        return
    
    # Load test audio
    audio_waveform = torch.randn(16000 * 3)  # 3 seconds of dummy audio
    prompt = "Based on this audio, describe the emotional state of the speaker in Chinese."
    
    # Generate blocks and check uncertainty values
    try:
        blocks_with_uncertainty, edge_latency_metrics = edge_model.generate_draft_blocks(
            audio_features=audio_waveform,
            prompt=prompt,
            block_size=4,
            max_blocks=6,
            temperature=0.7,
            top_p=0.9
        )
        
        logger.info(f"\nGenerated {len(blocks_with_uncertainty)} blocks")
        logger.info("=" * 60)
        
        for i, block in enumerate(blocks_with_uncertainty):
            uncertainty = block.get('uncertainty_signals', {})
            entropy = uncertainty.get('entropy', [])
            margin = uncertainty.get('margin', [])
            token_log_probs = uncertainty.get('token_log_probs', [])
            
            logger.info(f"Block {i}:")
            logger.info(f"  Text: {block.get('text', 'N/A')}")
            logger.info(f"  Tokens: {len(block.get('tokens', []))}")
            logger.info(f"  Entropy: {entropy}")
            logger.info(f"  Margin: {margin}")
            logger.info(f"  Token log probs: {token_log_probs}")
            logger.info(f"  Should verify: {block.get('should_verify', False)}")
            
            # Check if entropy is a list or single value
            if isinstance(entropy, list) and len(entropy) > 0:
                avg_entropy = sum(entropy) / len(entropy)
                max_entropy = max(entropy)
                logger.info(f"  Average entropy: {avg_entropy:.3f}")
                logger.info(f"  Max entropy: {max_entropy:.3f}")
            elif isinstance(entropy, (int, float)):
                logger.info(f"  Single entropy value: {entropy:.3f}")
            
            logger.info("-" * 40)
        
        # Test different thresholds
        logger.info("\nTesting different verification thresholds:")
        thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        
        for threshold in thresholds:
            selected_blocks = []
            for block in blocks_with_uncertainty:
                uncertainty = block.get('uncertainty_signals', {})
                entropy = uncertainty.get('entropy', [0])[0] if isinstance(uncertainty.get('entropy', []), list) else uncertainty.get('entropy', 0)
                
                if entropy > threshold:
                    selected_blocks.append(block)
            
            logger.info(f"  Threshold {threshold}: {len(selected_blocks)} blocks selected")
        
    except Exception as e:
        logger.error(f"Failed to generate blocks: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    debug_uncertainty_values()
