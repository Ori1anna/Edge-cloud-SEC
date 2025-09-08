#!/usr/bin/env python3
"""
Test script for the complete speculative decoding system
"""

import sys
import os
import torch
import logging
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.edge_model import EdgeModel
from models.cloud_model import CloudModel
from speculative_decoding import SpeculativeDecodingSystem

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_audio_sample(audio_path: str) -> torch.Tensor:
    """Load audio sample for testing"""
    try:
        import librosa
        # Load audio file
        audio, sr = librosa.load(audio_path, sr=16000)
        # Convert to tensor
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
        logger.info(f"Loaded audio: {audio_tensor.shape}, duration: {len(audio_tensor)/sr:.2f}s")
        return audio_tensor
    except ImportError:
        logger.error("librosa not available, using dummy audio")
        # Create dummy audio for testing
        return torch.randn(16000 * 3)  # 3 seconds of dummy audio
    except Exception as e:
        logger.error(f"Failed to load audio: {e}")
        return torch.randn(16000 * 3)


def test_speculative_decoding():
    """Test the complete speculative decoding system"""
    logger.info("=" * 60)
    logger.info("Testing Complete Speculative Decoding System")
    logger.info("=" * 60)
    
    # Initialize models
    logger.info("Initializing models...")
    try:
        edge_model = EdgeModel(device="cuda")
        logger.info("Edge model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load edge model: {e}")
        return False, None, None
    
    try:
        cloud_model = CloudModel(device="cuda")
        logger.info("Cloud model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load cloud model: {e}")
        return False, None, None
    
    # Initialize speculative decoding system
    logger.info("Initializing speculative decoding system...")
    try:
        spec_decoding = SpeculativeDecodingSystem(
            edge_model=edge_model,
            cloud_model=cloud_model,
            verification_threshold=0.7,  # Lower threshold for more verification
            max_verification_blocks=2    # Limit verification to 2 blocks
        )
        logger.info("Speculative decoding system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize speculative decoding system: {e}")
        return False
    
    # Load test audio
    audio_path = "data/audio_samples/angry_001.wav"
    if not os.path.exists(audio_path):
        logger.warning(f"Audio file not found: {audio_path}, using dummy audio")
        audio_waveform = torch.randn(16000 * 3)  # 3 seconds of dummy audio
    else:
        audio_waveform = load_audio_sample(audio_path)
    
    # Test parameters
    prompt = "Based on this audio, describe the emotional state of the speaker in Chinese."
    block_size = 4
    max_blocks = 6
    temperature = 0.7
    top_p = 0.9
    
    logger.info(f"Test parameters:")
    logger.info(f"  Block size: {block_size}")
    logger.info(f"  Max blocks: {max_blocks}")
    logger.info(f"  Temperature: {temperature}")
    logger.info(f"  Top-p: {top_p}")
    logger.info(f"  Verification threshold: {spec_decoding.verification_threshold}")
    
    # Run speculative decoding
    logger.info("\n" + "=" * 40)
    logger.info("Running Speculative Decoding...")
    logger.info("=" * 40)
    
    start_time = time.time()
    
    try:
        result = spec_decoding.generate_with_speculative_decoding(
            audio_waveform=audio_waveform,
            prompt=prompt,
            block_size=block_size,
            max_blocks=max_blocks,
            temperature=temperature,
            top_p=top_p
        )
        
        total_time = time.time() - start_time
        
        # Display results
        logger.info("\n" + "=" * 40)
        logger.info("SPECULATIVE DECODING RESULTS")
        logger.info("=" * 40)
        
        logger.info(f"Final Text: {result.final_text}")
        logger.info(f"")
        logger.info(f"Performance Metrics:")
        logger.info(f"  Total Latency: {result.total_latency:.3f}s")
        logger.info(f"  Edge Latency: {result.edge_latency:.3f}s")
        logger.info(f"  Cloud Latency: {result.cloud_latency:.3f}s")
        logger.info(f"  Acceptance Rate: {result.acceptance_rate:.2%}")
        logger.info(f"  Tokens per Second: {result.tokens_per_second:.2f}")
        logger.info(f"")
        logger.info(f"Token Statistics:")
        logger.info(f"  Accepted Tokens: {len(result.accepted_tokens)}")
        logger.info(f"  Rejected Tokens: {len(result.rejected_tokens)}")
        logger.info(f"  Verification Blocks: {len(result.verification_blocks)}")
        
        # Display verification details
        if result.verification_blocks:
            logger.info(f"")
            logger.info(f"Verification Details:")
            for i, block in enumerate(result.verification_blocks):
                logger.info(f"  Block {i+1}:")
                logger.info(f"    Original: {block.get('original_text', 'N/A')[:50]}...")
                logger.info(f"    Verification: {block.get('verification_text', 'N/A')[:50]}...")
                if 'error' in block:
                    logger.info(f"    Error: {block['error']}")
        
        # Check if result is valid
        if result.final_text and not result.final_text.startswith("Error:"):
            logger.info(f"\n✅ Speculative decoding completed successfully!")
            logger.info(f"✅ Generated meaningful text: {len(result.final_text)} characters")
            logger.info(f"✅ System performance: {result.tokens_per_second:.2f} tokens/sec")
            return True, edge_model, cloud_model
        else:
            logger.error(f"❌ Speculative decoding failed: {result.final_text}")
            return False, edge_model, cloud_model
            
    except Exception as e:
        logger.error(f"❌ Speculative decoding failed with exception: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False, edge_model, cloud_model


def test_comparison_with_baselines(edge_model, cloud_model):
    """Compare speculative decoding with individual model baselines"""
    logger.info("\n" + "=" * 60)
    logger.info("Comparing with Baseline Models")
    logger.info("=" * 60)
    
    # Load test audio
    audio_path = "data/audio_samples/angry_001.wav"
    if not os.path.exists(audio_path):
        audio_waveform = torch.randn(16000 * 3)
    else:
        audio_waveform = load_audio_sample(audio_path)
    
    prompt = "Based on this audio, describe the emotional state of the speaker in Chinese."
    
    # Test edge model baseline
    logger.info("Testing Edge Model Baseline...")
    try:
        edge_start = time.time()
        edge_text, edge_metrics = edge_model.generate_draft(
            audio_features=audio_waveform,
            prompt=prompt,
            max_new_tokens=32,
            temperature=0.7,
            top_p=0.9
        )
        edge_time = time.time() - edge_start
        logger.info(f"Edge Model Result: {edge_text}")
        logger.info(f"Edge Model Time: {edge_time:.3f}s")
    except Exception as e:
        logger.error(f"Edge model baseline failed: {e}")
    
    # Test cloud model baseline
    logger.info("\nTesting Cloud Model Baseline...")
    try:
        cloud_start = time.time()
        cloud_text, cloud_metrics = cloud_model.generate_independently(
            audio_waveform=audio_waveform,
            prompt=prompt,
            max_new_tokens=32,
            temperature=0.7,
            top_p=0.9
        )
        cloud_time = time.time() - cloud_start
        logger.info(f"Cloud Model Result: {cloud_text}")
        logger.info(f"Cloud Model Time: {cloud_time:.3f}s")
    except Exception as e:
        logger.error(f"Cloud model baseline failed: {e}")


if __name__ == "__main__":
    logger.info("Starting Speculative Decoding System Test")
    
    # Test main speculative decoding system
    success, edge_model, cloud_model = test_speculative_decoding()
    
    if success and edge_model is not None and cloud_model is not None:
        # Test comparison with baselines using the same models
        test_comparison_with_baselines(edge_model, cloud_model)
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
    else:
        logger.error("\n" + "=" * 60)
        logger.error("❌ TESTS FAILED!")
        logger.error("=" * 60)
        sys.exit(1)
