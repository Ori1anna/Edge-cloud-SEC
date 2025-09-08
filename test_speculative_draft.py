#!/usr/bin/env python3
"""
Test script for speculative decoding draft generation functionality
"""

import os
import sys
import torch
import logging
import json
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.models.edge_model import EdgeModel
from src.data.audio_processor import AudioProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_block_generation():
    """Test the new block-based draft generation"""
    
    # Initialize edge model
    logger.info("Initializing EdgeModel...")
    edge_model = EdgeModel(
        model_name="Qwen/Qwen2.5-Omni-3B",
        device="cuda",
        dtype="float16"
    )
    
    # Load a test audio file
    test_audio_path = "data/processed/secap/wav_16k/tx_emotion_00201000015.wav"
    if not os.path.exists(test_audio_path):
        logger.error(f"Test audio file not found: {test_audio_path}")
        return
    
    logger.info(f"Loading test audio: {test_audio_path}")
    audio_processor = AudioProcessor()
    audio_features = audio_processor.load_audio(test_audio_path)
    
    # First test the original generate_draft method to ensure it still works
    logger.info("Testing original generate_draft method...")
    original_text, original_metrics = edge_model.generate_draft(
        audio_features=audio_features,
        prompt="基于这个音频，用中文描述说话人的情感状态。",
        max_new_tokens=20,
        temperature=0.7,
        top_p=0.9
    )
    logger.info(f"Original method result: '{original_text}'")
    logger.info(f"Original metrics: {original_metrics}")
    
    # Test block generation
    logger.info("Testing block-based generation...")
    blocks, latency_metrics = edge_model.generate_draft_blocks(
        audio_features=audio_features,
        prompt="基于这个音频，用中文描述说话人的情感状态。",
        block_size=5,  # Larger blocks for testing
        max_blocks=3,  # Limit blocks for testing
        temperature=0.7,
        top_p=0.9
    )
    
    # Print results
    logger.info(f"Generated {len(blocks)} blocks")
    logger.info(f"Latency metrics: {latency_metrics}")
    
    print("\n" + "="*50)
    print("BLOCK GENERATION RESULTS")
    print("="*50)
    
    for i, block in enumerate(blocks):
        print(f"\nBlock {i+1}:")
        print(f"  Text: '{block['text']}'")
        print(f"  Tokens: {block['tokens']}")
        
        # Decode individual tokens for debugging
        if block['tokens']:
            token_texts = []
            for token in block['tokens']:
                try:
                    token_text = edge_model.processor.tokenizer.decode([token], skip_special_tokens=False)
                    token_texts.append(f"{token}:'{token_text}'")
                except:
                    token_texts.append(f"{token}:<unknown>")
            print(f"  Token details: {token_texts}")
        
        print(f"  Should verify: {block['should_verify']}")
        print(f"  Generation time: {block['block_generation_time']:.3f}s")
        
        if 'uncertainty_signals' in block:
            uncertainty = block['uncertainty_signals']
            if 'entropy' in uncertainty and uncertainty['entropy']:
                if isinstance(uncertainty['entropy'], list):
                    avg_entropy = sum(uncertainty['entropy']) / len(uncertainty['entropy'])
                else:
                    avg_entropy = uncertainty['entropy']
                print(f"  Avg entropy: {avg_entropy:.3f}")
            
            if 'margin' in uncertainty and uncertainty['margin']:
                if isinstance(uncertainty['margin'], list):
                    avg_margin = sum(uncertainty['margin']) / len(uncertainty['margin'])
                else:
                    avg_margin = uncertainty['margin']
                print(f"  Avg margin: {avg_margin:.3f}")
            
            if 'token_log_probs' in uncertainty and uncertainty['token_log_probs']:
                if isinstance(uncertainty['token_log_probs'], list):
                    # Handle nested lists (list of lists)
                    if uncertainty['token_log_probs'] and isinstance(uncertainty['token_log_probs'][0], list):
                        # Flatten the nested list
                        flat_probs = [prob for sublist in uncertainty['token_log_probs'] for prob in sublist]
                        avg_log_prob = sum(flat_probs) / len(flat_probs) if flat_probs else 0.0
                    else:
                        avg_log_prob = sum(uncertainty['token_log_probs']) / len(uncertainty['token_log_probs'])
                else:
                    avg_log_prob = uncertainty['token_log_probs']
                print(f"  Avg log prob: {avg_log_prob:.3f}")
    
    # Test content pattern detection
    print("\n" + "="*50)
    print("CONTENT PATTERN DETECTION TEST")
    print("="*50)
    
    test_texts = [
        "悲伤，情绪低落",
        "情绪不明确，情绪不明确，情绪不明确",
        "2024年1月1日",
        "John Smith",
        "a",  # Very short
        "这是一个非常长的文本，包含了很多重复的词汇和短语，用来测试内容模式检测功能是否能够正确识别出这种重复模式"
    ]
    
    for text in test_texts:
        patterns = edge_model.detect_content_patterns(text)
        print(f"\nText: '{text}'")
        for pattern, detected in patterns.items():
            if detected:
                print(f"  {pattern}: {detected}")
    
    # Save results
    results = {
        'blocks': blocks,
        'latency_metrics': latency_metrics,
        'test_audio_path': test_audio_path
    }
    
    output_path = "test_speculative_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to: {output_path}")

if __name__ == "__main__":
    test_block_generation()
