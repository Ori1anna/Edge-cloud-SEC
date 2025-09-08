#!/usr/bin/env python3
"""
Test script to verify block generation on multiple audio samples
"""

import os
import sys
import torch
import logging
import random
import time
from typing import List, Dict, Any

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.models.edge_model import EdgeModel
from src.data.audio_processor import AudioProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_audio_files(audio_dir: str, max_files: int = 10) -> List[str]:
    """Get a random sample of audio files for testing"""
    if not os.path.exists(audio_dir):
        logger.error(f"Audio directory not found: {audio_dir}")
        return []
    
    all_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    if not all_files:
        logger.error(f"No .wav files found in {audio_dir}")
        return []
    
    # Randomly sample files
    sample_files = random.sample(all_files, min(max_files, len(all_files)))
    return [os.path.join(audio_dir, f) for f in sample_files]

def test_single_sample(edge_model: EdgeModel, audio_processor: AudioProcessor, 
                      audio_path: str, sample_idx: int) -> Dict[str, Any]:
    """Test block generation on a single audio sample"""
    logger.info(f"Testing sample {sample_idx}: {os.path.basename(audio_path)}")
    
    try:
        # Load audio
        audio_features = audio_processor.load_audio(audio_path)
        
        # Generate blocks
        start_time = time.time()
        blocks, latency_metrics = edge_model.generate_draft_blocks(
            audio_features=audio_features,
            prompt="基于这个音频，用中文描述说话人的情感状态。",
            block_size=5,
            max_blocks=6,  # Reduced for faster testing
            temperature=0.7,
            top_p=0.9
        )
        total_time = time.time() - start_time
        
        # Analyze results
        total_tokens = sum(len(block['tokens']) for block in blocks)
        total_blocks = len(blocks)
        
        # Calculate average entropy
        all_entropies = []
        verify_count = 0
        for block in blocks:
            if 'uncertainty_signals' in block and 'entropy' in block['uncertainty_signals']:
                entropy = block['uncertainty_signals']['entropy']
                if isinstance(entropy, list):
                    all_entropies.extend(entropy)
                else:
                    all_entropies.append(entropy)
            
            if block.get('should_verify', False):
                verify_count += 1
        
        avg_entropy = sum(all_entropies) / len(all_entropies) if all_entropies else 0.0
        
        # Get generated text
        full_text = ""
        for block in blocks:
            full_text += block['text']
        
        return {
            'sample_idx': sample_idx,
            'audio_file': os.path.basename(audio_path),
            'total_blocks': total_blocks,
            'total_tokens': total_tokens,
            'avg_entropy': avg_entropy,
            'verify_ratio': verify_count / total_blocks if total_blocks > 0 else 0.0,
            'total_time': total_time,
            'tokens_per_second': total_tokens / total_time if total_time > 0 else 0.0,
            'full_text': full_text,
            'blocks': blocks,
            'latency_metrics': latency_metrics,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Error testing sample {sample_idx}: {e}")
        return {
            'sample_idx': sample_idx,
            'audio_file': os.path.basename(audio_path),
            'error': str(e),
            'success': False
        }

def test_multiple_samples():
    """Test block generation on multiple audio samples"""
    
    # Initialize models
    logger.info("Initializing EdgeModel...")
    edge_model = EdgeModel(
        model_name="Qwen/Qwen2.5-Omni-3B",
        device="cuda",
        dtype="float16"
    )
    
    audio_processor = AudioProcessor()
    
    # Get audio files
    audio_dir = "data/processed/secap/wav_16k"
    audio_files = get_audio_files(audio_dir, max_files=8)  # Test 8 samples
    
    if not audio_files:
        logger.error("No audio files found for testing")
        return
    
    logger.info(f"Testing {len(audio_files)} audio samples...")
    
    # Test each sample
    results = []
    for i, audio_path in enumerate(audio_files):
        result = test_single_sample(edge_model, audio_processor, audio_path, i + 1)
        results.append(result)
        
        # Print sample result
        if result['success']:
            print(f"\n{'='*60}")
            print(f"SAMPLE {result['sample_idx']}: {result['audio_file']}")
            print(f"{'='*60}")
            print(f"Blocks: {result['total_blocks']}, Tokens: {result['total_tokens']}")
            print(f"Avg Entropy: {result['avg_entropy']:.3f}")
            print(f"Verify Ratio: {result['verify_ratio']:.2f}")
            print(f"Time: {result['total_time']:.2f}s, Speed: {result['tokens_per_second']:.1f} tok/s")
            print(f"Generated Text: {result['full_text'][:100]}{'...' if len(result['full_text']) > 100 else ''}")
        else:
            print(f"\n{'='*60}")
            print(f"SAMPLE {result['sample_idx']}: {result['audio_file']} - FAILED")
            print(f"Error: {result['error']}")
    
    # Summary statistics
    successful_results = [r for r in results if r['success']]
    if successful_results:
        print(f"\n{'='*80}")
        print("SUMMARY STATISTICS")
        print(f"{'='*80}")
        
        total_blocks = sum(r['total_blocks'] for r in successful_results)
        total_tokens = sum(r['total_tokens'] for r in successful_results)
        total_time = sum(r['total_time'] for r in successful_results)
        avg_entropies = [r['avg_entropy'] for r in successful_results]
        verify_ratios = [r['verify_ratio'] for r in successful_results]
        
        print(f"Successful samples: {len(successful_results)}/{len(results)}")
        print(f"Total blocks generated: {total_blocks}")
        print(f"Total tokens generated: {total_tokens}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average tokens per second: {total_tokens/total_time:.1f}")
        print(f"Average entropy: {sum(avg_entropies)/len(avg_entropies):.3f}")
        print(f"Average verify ratio: {sum(verify_ratios)/len(verify_ratios):.2f}")
        print(f"Entropy range: {min(avg_entropies):.3f} - {max(avg_entropies):.3f}")
        print(f"Verify ratio range: {min(verify_ratios):.2f} - {max(verify_ratios):.2f}")
        
        # Show some example texts
        print(f"\n{'='*80}")
        print("EXAMPLE GENERATED TEXTS")
        print(f"{'='*80}")
        for i, result in enumerate(successful_results[:3]):
            print(f"\nSample {result['sample_idx']} ({result['audio_file']}):")
            print(f"Text: {result['full_text']}")
    
    else:
        print(f"\n{'='*60}")
        print("NO SUCCESSFUL SAMPLES")
        print(f"{'='*60}")

if __name__ == "__main__":
    test_multiple_samples()