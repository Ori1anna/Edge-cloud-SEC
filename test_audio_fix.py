#!/usr/bin/env python3
"""
Test script to verify audio input fix
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

def test_audio_fix():
    """Test if audio input is now working correctly"""
    
    # Initialize models
    logger.info("Initializing EdgeModel...")
    edge_model = EdgeModel(
        model_name="Qwen/Qwen2.5-Omni-3B",
        device="cuda",
        dtype="float16"
    )
    
    audio_processor = AudioProcessor()
    
    # Test with one audio file
    audio_path = "data/processed/secap/wav_16k/tx_emotion_00201000015.wav"
    logger.info(f"Testing with audio: {audio_path}")
    
    # Load audio
    audio_features = audio_processor.load_audio(audio_path)
    
    # Test block generation
    logger.info("Testing block generation...")
    blocks, latency_metrics = edge_model.generate_draft_blocks(
        audio_features=audio_features,
        prompt="基于这个音频，用中文描述说话人的情感状态。",
        block_size=3,  # Small blocks for quick test
        max_blocks=2,  # Just 2 blocks for quick test
        temperature=0.7,
        top_p=0.9
    )
    
    # Print results
    print(f"\n{'='*60}")
    print("AUDIO FIX TEST RESULTS")
    print(f"{'='*60}")
    
    print(f"Total blocks generated: {len(blocks)}")
    print(f"Total tokens: {sum(len(block['tokens']) for block in blocks)}")
    
    full_text = ""
    for i, block in enumerate(blocks):
        print(f"\nBlock {i+1}:")
        print(f"  Text: '{block['text']}'")
        print(f"  Should verify: {block['should_verify']}")
        if 'uncertainty_signals' in block and 'entropy' in block['uncertainty_signals']:
            entropy = block['uncertainty_signals']['entropy']
            if isinstance(entropy, list):
                avg_entropy = sum(entropy) / len(entropy)
            else:
                avg_entropy = entropy
            print(f"  Avg entropy: {avg_entropy:.3f}")
        full_text += block['text']
    
    print(f"\nFull generated text: {full_text}")
    
    # Check if model actually heard the audio
    if "没有" in full_text or "听不到" in full_text or "没有提供" in full_text:
        print(f"\n❌ ISSUE: Model still says it can't hear audio")
    else:
        print(f"\n✅ SUCCESS: Model appears to be processing audio correctly")

if __name__ == "__main__":
    test_audio_fix()

