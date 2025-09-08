#!/usr/bin/env python3
"""
Debug script to check audio loading and processing
"""

import os
import sys
import torch
import logging

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.data.audio_processor import AudioProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_audio_loading():
    """Debug audio loading process"""
    
    audio_processor = AudioProcessor()
    
    # Test with a few audio files
    test_files = [
        "data/processed/secap/wav_16k/tx_emotion_00201000015.wav",
        "data/processed/secap/wav_16k/tx_emotion_00210000476.wav",
        "data/processed/secap/wav_16k/tx_emotion_00305000009.wav"
    ]
    
    for audio_path in test_files:
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            continue
            
        logger.info(f"Testing audio: {audio_path}")
        
        try:
            # Load audio
            audio_features = audio_processor.load_audio(audio_path)
            
            # Check audio properties
            logger.info(f"  Audio shape: {audio_features.shape}")
            logger.info(f"  Audio dtype: {audio_features.dtype}")
            logger.info(f"  Audio min/max: {audio_features.min():.4f} / {audio_features.max():.4f}")
            logger.info(f"  Audio mean: {audio_features.mean():.4f}")
            logger.info(f"  Audio std: {audio_features.std():.4f}")
            logger.info(f"  Non-zero samples: {(audio_features != 0).sum().item()}")
            logger.info(f"  Total samples: {audio_features.numel()}")
            
            # Check if audio is silent
            if audio_features.abs().max() < 1e-6:
                logger.warning(f"  WARNING: Audio appears to be silent!")
            else:
                logger.info(f"  Audio has content (not silent)")
                
        except Exception as e:
            logger.error(f"  Error loading audio: {e}")

if __name__ == "__main__":
    debug_audio_loading()

