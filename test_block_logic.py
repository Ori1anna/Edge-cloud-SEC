#!/usr/bin/env python3
"""
Test script to verify the new block generation logic
"""

import os
import sys
import torch
import logging

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.models.edge_model import EdgeModel
from src.data.audio_processor import AudioProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_block_logic():
    """Test the new block generation logic"""
    
    # Initialize models
    logger.info("Initializing EdgeModel...")
    edge_model = EdgeModel(
        model_name="Qwen/Qwen2.5-Omni-3B",
        device="cuda",
        dtype="float16"
    )
    
    audio_processor = AudioProcessor()
    
    # Load test audio
    test_audio_path = "data/processed/secap/wav_16k/tx_emotion_00201000015.wav"
    if not os.path.exists(test_audio_path):
        logger.error(f"Test audio file not found: {test_audio_path}")
        return
    
    logger.info(f"Loading test audio: {test_audio_path}")
    audio_features = audio_processor.load_audio(test_audio_path)
    
    # Test block generation with strict block size
    logger.info("Testing block-based generation with block_size=5...")
    blocks, latency_metrics = edge_model.generate_draft_blocks(
        audio_features=audio_features,
        prompt="基于这个音频，用中文描述说话人的情感状态。",
        block_size=5,  # Exactly 5 tokens per block
        max_blocks=8,  # Maximum 8 blocks to test more generation
        temperature=0.7,
        top_p=0.9
    )
    
    # Print detailed results
    print("\n" + "="*60)
    print("BLOCK GENERATION RESULTS (block_size=5)")
    print("="*60)
    
    print(f"Total blocks generated: {len(blocks)}")
    print(f"Total tokens: {sum(len(block['tokens']) for block in blocks)}")
    
    for i, block in enumerate(blocks):
        print(f"\nBlock {i+1}:")
        print(f"  Tokens count: {len(block['tokens'])}")
        print(f"  Tokens: {block['tokens']}")
        print(f"  Text: '{block['text']}'")
        print(f"  Should verify: {block['should_verify']}")
        print(f"  Generation time: {block['block_generation_time']:.3f}s")
        
        # Show individual token details
        if block['tokens']:
            token_texts = []
            for token in block['tokens']:
                try:
                    token_text = edge_model.processor.tokenizer.decode([token], skip_special_tokens=False)
                    token_texts.append(f"{token}:'{token_text}'")
                except:
                    token_texts.append(f"{token}:<unknown>")
            print(f"  Token details: {token_texts}")
        
        # Show uncertainty if available
        if 'uncertainty_signals' in block:
            uncertainty = block['uncertainty_signals']
            if 'entropy' in uncertainty and uncertainty['entropy']:
                if isinstance(uncertainty['entropy'], list):
                    avg_entropy = sum(uncertainty['entropy']) / len(uncertainty['entropy'])
                else:
                    avg_entropy = uncertainty['entropy']
                print(f"  Avg entropy: {avg_entropy:.3f}")
    
    print(f"\nLatency metrics: {latency_metrics}")

if __name__ == "__main__":
    test_block_logic()
