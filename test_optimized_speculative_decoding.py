#!/usr/bin/env python3
"""
Test script for the optimized speculative decoding system
Compares original vs optimized versions
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
from optimized_speculative_decoding import OptimizedSpeculativeDecodingSystem

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


def test_original_speculative_decoding(edge_model, cloud_model, audio_waveform, prompt):
    """Test the original speculative decoding system"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing ORIGINAL Speculative Decoding System")
    logger.info("=" * 60)
    
    # Initialize original speculative decoding system
    try:
        original_spec_decoding = SpeculativeDecodingSystem(
            edge_model=edge_model,
            cloud_model=cloud_model,
            verification_threshold=0.7,
            max_verification_blocks=2
        )
        logger.info("Original speculative decoding system initialized")
    except Exception as e:
        logger.error(f"Failed to initialize original system: {e}")
        return None
    
    # Test parameters
    block_size = 4
    max_blocks = 6
    temperature = 0.7
    top_p = 0.9
    
    logger.info(f"Test parameters: block_size={block_size}, max_blocks={max_blocks}")
    
    # Run original speculative decoding
    start_time = time.time()
    
    try:
        result = original_spec_decoding.generate_with_speculative_decoding(
            audio_waveform=audio_waveform,
            prompt=prompt,
            block_size=block_size,
            max_blocks=max_blocks,
            temperature=temperature,
            top_p=top_p
        )
        
        total_time = time.time() - start_time
        
        # Display results
        logger.info(f"\n📊 ORIGINAL SYSTEM RESULTS:")
        logger.info(f"  Final Text: {result.final_text[:100]}...")
        logger.info(f"  Total Latency: {result.total_latency:.3f}s")
        logger.info(f"  Edge Latency: {result.edge_latency:.3f}s")
        logger.info(f"  Cloud Latency: {result.cloud_latency:.3f}s")
        logger.info(f"  Acceptance Rate: {result.acceptance_rate:.2%}")
        logger.info(f"  Tokens per Second: {result.tokens_per_second:.2f}")
        logger.info(f"  Accepted Tokens: {len(result.accepted_tokens)}")
        logger.info(f"  Rejected Tokens: {len(result.rejected_tokens)}")
        logger.info(f"  Verification Blocks: {len(result.verification_blocks)}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Original speculative decoding failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


def test_optimized_speculative_decoding(edge_model, cloud_model, audio_waveform, prompt):
    """Test the optimized speculative decoding system"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing OPTIMIZED Speculative Decoding System")
    logger.info("=" * 60)
    
    # Initialize optimized speculative decoding system
    try:
        optimized_spec_decoding = OptimizedSpeculativeDecodingSystem(
            edge_model=edge_model,
            cloud_model=cloud_model,
            verification_threshold=0.8,  # Higher threshold for fewer verifications
            max_verification_blocks=2
        )
        logger.info("Optimized speculative decoding system initialized")
    except Exception as e:
        logger.error(f"Failed to initialize optimized system: {e}")
        return None
    
    # Test parameters (same as original for fair comparison)
    block_size = 4
    max_blocks = 6
    temperature = 0.7
    top_p = 0.9
    
    logger.info(f"Test parameters: block_size={block_size}, max_blocks={max_blocks}")
    
    # Run optimized speculative decoding
    start_time = time.time()
    
    try:
        result = optimized_spec_decoding.generate_with_optimized_speculative_decoding(
            audio_waveform=audio_waveform,
            prompt=prompt,
            block_size=block_size,
            max_blocks=max_blocks,
            temperature=temperature,
            top_p=top_p
        )
        
        total_time = time.time() - start_time
        
        # Display results
        logger.info(f"\n📊 OPTIMIZED SYSTEM RESULTS:")
        logger.info(f"  Final Text: {result.final_text[:100]}...")
        logger.info(f"  Total Latency: {result.total_latency:.3f}s")
        logger.info(f"  Edge Latency: {result.edge_latency:.3f}s")
        logger.info(f"  Cloud Latency: {result.cloud_latency:.3f}s")
        logger.info(f"  Acceptance Rate: {result.acceptance_rate:.2%}")
        logger.info(f"  Tokens per Second: {result.tokens_per_second:.2f}")
        logger.info(f"  Speedup Ratio: {result.speedup_ratio:.2f}x")
        logger.info(f"  Accepted Tokens: {len(result.accepted_tokens)}")
        logger.info(f"  Rejected Tokens: {len(result.rejected_tokens)}")
        logger.info(f"  Verification Blocks: {result.latency_metrics.get('verification_blocks', 0)}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Optimized speculative decoding failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


def compare_results(original_result, optimized_result):
    """Compare results between original and optimized systems"""
    logger.info("\n" + "=" * 60)
    logger.info("PERFORMANCE COMPARISON")
    logger.info("=" * 60)
    
    if original_result is None or optimized_result is None:
        logger.error("Cannot compare - one or both results are None")
        return
    
    # Calculate improvements
    latency_improvement = ((original_result.total_latency - optimized_result.total_latency) / original_result.total_latency) * 100
    cloud_latency_improvement = ((original_result.cloud_latency - optimized_result.cloud_latency) / original_result.cloud_latency) * 100 if original_result.cloud_latency > 0 else 0
    tps_improvement = ((optimized_result.tokens_per_second - original_result.tokens_per_second) / original_result.tokens_per_second) * 100
    
    logger.info(f"📈 PERFORMANCE IMPROVEMENTS:")
    logger.info(f"  Total Latency: {original_result.total_latency:.3f}s → {optimized_result.total_latency:.3f}s ({latency_improvement:+.1f}%)")
    logger.info(f"  Cloud Latency: {original_result.cloud_latency:.3f}s → {optimized_result.cloud_latency:.3f}s ({cloud_latency_improvement:+.1f}%)")
    logger.info(f"  Tokens/sec: {original_result.tokens_per_second:.2f} → {optimized_result.tokens_per_second:.2f} ({tps_improvement:+.1f}%)")
    logger.info(f"  Acceptance Rate: {original_result.acceptance_rate:.2%} → {optimized_result.acceptance_rate:.2%}")
    
    logger.info(f"\n📊 DETAILED COMPARISON:")
    logger.info(f"  Original Verification Blocks: {len(original_result.verification_blocks)}")
    logger.info(f"  Optimized Verification Blocks: {optimized_result.latency_metrics.get('verification_blocks', 0)}")
    logger.info(f"  Original Accepted Tokens: {len(original_result.accepted_tokens)}")
    logger.info(f"  Optimized Accepted Tokens: {len(optimized_result.accepted_tokens)}")
    
    # Determine if optimization was successful
    if latency_improvement > 0:
        logger.info(f"\n✅ OPTIMIZATION SUCCESSFUL!")
        logger.info(f"✅ Achieved {latency_improvement:.1f}% latency reduction")
        if optimized_result.speedup_ratio > 1.0:
            logger.info(f"✅ Achieved {optimized_result.speedup_ratio:.2f}x speedup over cloud-only baseline")
    else:
        logger.info(f"\n⚠️  OPTIMIZATION NEEDS IMPROVEMENT")
        logger.info(f"⚠️  Latency increased by {abs(latency_improvement):.1f}%")
    
    # Text quality comparison
    logger.info(f"\n📝 TEXT QUALITY COMPARISON:")
    logger.info(f"  Original Text Length: {len(original_result.final_text)} characters")
    logger.info(f"  Optimized Text Length: {len(optimized_result.final_text)} characters")
    logger.info(f"  Original Text: {original_result.final_text[:150]}...")
    logger.info(f"  Optimized Text: {optimized_result.final_text[:150]}...")


def main():
    """Main test function"""
    logger.info("Starting Optimized Speculative Decoding System Test")
    
    # Initialize models (load once to avoid duplication)
    logger.info("Initializing models...")
    try:
        edge_model = EdgeModel(device="cuda")
        logger.info("Edge model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load edge model: {e}")
        return False
    
    try:
        cloud_model = CloudModel(device="cuda")
        logger.info("Cloud model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load cloud model: {e}")
        return False
    
    # Load test audio
    audio_path = "data/audio_samples/angry_001.wav"
    if not os.path.exists(audio_path):
        logger.warning(f"Audio file not found: {audio_path}, using dummy audio")
        audio_waveform = torch.randn(16000 * 3)  # 3 seconds of dummy audio
    else:
        audio_waveform = load_audio_sample(audio_path)
    
    prompt = "Based on this audio, describe the emotional state of the speaker in Chinese."
    
    # Test original system
    original_result = test_original_speculative_decoding(edge_model, cloud_model, audio_waveform, prompt)
    
    # Test optimized system
    optimized_result = test_optimized_speculative_decoding(edge_model, cloud_model, audio_waveform, prompt)
    
    # Compare results
    compare_results(original_result, optimized_result)
    
    # Final summary
    if original_result and optimized_result:
        logger.info("\n" + "=" * 60)
        logger.info("🎉 COMPARISON TEST COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        return True
    else:
        logger.error("\n" + "=" * 60)
        logger.error("❌ COMPARISON TEST FAILED!")
        logger.error("=" * 60)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
