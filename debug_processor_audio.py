#!/usr/bin/env python3
"""
Debug script to check processor audio handling
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

def debug_processor_audio():
    """Debug processor audio handling"""
    
    # Initialize model
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
    logger.info(f"Audio shape: {audio_features.shape}")
    logger.info(f"Audio range: {audio_features.min():.4f} to {audio_features.max():.4f}")
    
    # Test conversation format
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
                {"type": "text", "text": "基于这个音频，用中文描述说话人的情感状态。"}
            ],
        },
    ]
    
    # Apply chat template
    text = edge_model.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    logger.info(f"Chat template result length: {len(text)}")
    logger.info(f"Chat template preview: {text[:200]}...")
    
    # Process inputs
    logger.info("Processing inputs with processor...")
    inputs = edge_model.processor(
        text=text, 
        audio=audio_features,
        return_tensors="pt", 
        padding=True
    )
    
    logger.info("Input keys:", list(inputs.keys()))
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            logger.info(f"  {key}: type={type(value)}")
    
    # Check if audio is properly included
    if 'audio_values' in inputs:
        audio_values = inputs['audio_values']
        logger.info(f"Audio values shape: {audio_values.shape}")
        logger.info(f"Audio values range: {audio_values.min():.4f} to {audio_values.max():.4f}")
        logger.info(f"Audio values non-zero: {(audio_values != 0).sum().item()}")
    else:
        logger.warning("No 'audio_values' found in inputs!")
    
    if 'audio_attention_mask' in inputs:
        audio_mask = inputs['audio_attention_mask']
        logger.info(f"Audio attention mask shape: {audio_mask.shape}")
        logger.info(f"Audio attention mask sum: {audio_mask.sum().item()}")
    else:
        logger.warning("No 'audio_attention_mask' found in inputs!")

if __name__ == "__main__":
    debug_processor_audio()

