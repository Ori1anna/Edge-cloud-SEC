#!/usr/bin/env python3
"""
Debug script to compare input processing between original and block methods
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

def debug_input_comparison():
    """Compare input processing between original and block methods"""
    
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
    audio_features = audio_processor.load_audio(test_audio_path)
    
    prompt = "基于这个音频，用中文描述说话人的情感状态。"
    
    # Prepare conversation format (same as both methods)
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
                {"type": "text", "text": prompt}
            ],
        },
    ]
    
    # Apply chat template
    text = edge_model.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    
    # Process inputs
    inputs = edge_model.processor(
        text=text, 
        audio=audio_features,
        return_tensors="pt", 
        padding=True
    )
    inputs = inputs.to(edge_model.device).to(edge_model.model.dtype)
    
    print("="*60)
    print("INPUT COMPARISON DEBUG")
    print("="*60)
    
    print(f"Input keys: {list(inputs.keys())}")
    print(f"Input IDs shape: {inputs['input_ids'].shape}")
    print(f"Input IDs length: {inputs['input_ids'].shape[1]}")
    
    # Show the last few tokens of input
    input_tokens = inputs['input_ids'][0].tolist()
    print(f"Last 10 input tokens: {input_tokens[-10:]}")
    
    # Decode the last few tokens
    last_tokens_text = edge_model.processor.tokenizer.decode(input_tokens[-10:], skip_special_tokens=False)
    print(f"Last 10 tokens as text: '{last_tokens_text}'")
    
    # Test original method first
    print("\n" + "="*40)
    print("TESTING ORIGINAL METHOD")
    print("="*40)
    
    original_text, original_metrics = edge_model.generate_draft(
        audio_features=audio_features,
        prompt=prompt,
        max_new_tokens=20,
        temperature=0.7,
        top_p=0.9
    )
    
    print(f"Original method result: '{original_text}'")
    print(f"Original method tokens: {original_metrics.get('output_tokens', 0)}")
    
    # Now test what happens with the same inputs in block method
    print("\n" + "="*40)
    print("TESTING BLOCK METHOD INPUT")
    print("="*40)
    
    # Use the same inputs as original method
    with torch.no_grad():
        outputs = edge_model.model.generate(
            **inputs,
            max_new_tokens=1,  # Generate exactly 1 token
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=edge_model.processor.tokenizer.eos_token_id,
            return_dict_in_generate=False,
            output_scores=False,
            return_audio=False
        )
        
        # Extract the new token
        new_token = outputs[0][-1].item()
        token_text = edge_model.processor.tokenizer.decode([new_token], skip_special_tokens=False)
        
        print(f"First token generated: {new_token} -> '{token_text}'")
        
        # Check if it's EOS
        is_eos = (new_token == edge_model.processor.tokenizer.eos_token_id or 
                 new_token == edge_model.processor.tokenizer.convert_tokens_to_ids('<|im_end|>') or
                 new_token == edge_model.processor.tokenizer.convert_tokens_to_ids('<|endoftext|>'))
        
        print(f"Is EOS token: {is_eos}")

if __name__ == "__main__":
    debug_input_comparison()

