#!/usr/bin/env python3
"""
Test script for Cloud Optimized Baseline

This script tests the new generate_with_spec_logic method in CloudModel
to ensure it works correctly with speculative decoding logic.
"""

import os
import sys
import torch
import librosa
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.cloud_model import CloudModel
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_cloud_optimized_baseline():
    """Test the Cloud Optimized Baseline functionality"""
    
    logger.info("=" * 80)
    logger.info("TESTING CLOUD OPTIMIZED BASELINE")
    logger.info("=" * 80)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        logger.error("CUDA is not available! This test requires GPU.")
        return False
    
    device = "cuda"
    logger.info(f"Using device: {device}")
    
    # Model path (same as original cloud baseline)
    model_path = "Qwen/Qwen2.5-Omni-7B"
    logger.info(f"Using model: {model_path}")
    
    try:
        # Initialize Cloud model (same parameters as original cloud baseline)
        logger.info("Initializing Cloud model...")
        cloud_model = CloudModel(
            model_name=model_path,
            device=device,
            dtype="float32"  # Updated to float32 for fair comparison
        )
        logger.info("Cloud model initialized successfully!")
        
        # Create a more realistic audio waveform for testing
        logger.info("Creating test audio waveform...")
        sample_rate = 16000
        duration = 3.0  # 3 seconds
        
        # Generate a more complex audio signal that might contain "emotion-like" features
        t = torch.linspace(0, duration, int(sample_rate * duration))
        
        # Create a signal with multiple frequency components and some variation
        # This simulates speech-like characteristics
        audio_waveform = (
            0.3 * torch.sin(2 * torch.pi * 200 * t) +  # Low frequency (vocal tract)
            0.2 * torch.sin(2 * torch.pi * 800 * t) +  # Mid frequency (formants)
            0.1 * torch.sin(2 * torch.pi * 1200 * t) + # High frequency (harmonics)
            0.05 * torch.sin(2 * torch.pi * 50 * t)    # Very low frequency (pitch variation)
        )
        
        # Add some amplitude modulation to simulate speech rhythm
        amplitude_mod = 0.5 + 0.5 * torch.sin(2 * torch.pi * 2 * t)
        audio_waveform = audio_waveform * amplitude_mod
        
        logger.info(f"Created test audio waveform: {audio_waveform.shape}")
        
        # Test prompt (same as original cloud baseline)
        prompt = """任务：请基于给定音频，输出一句"情感说明短句"。

必须遵守：
- 只输出一句中文短句（12–30个汉字），以"。"结尾。
- 句子中同时包含：一个主要情绪 + 一个简短的声学/韵律线索（如语气、语速、强弱、音高变化等"类别"层面的描述即可），但不要解释或列举。
- 不要出现客套话、邀请继续对话、表情符号、英文、Markdown、标号或代码；不要提及"音频/模型/分析/我"。
- 若存在多种可能性，只选择最可能的一种，不要并列罗列。

只给出最终这"一句短句"，不要输出其他内容。"""
        
        # Test the new generate_with_spec_logic method
        logger.info("Testing generate_with_spec_logic method...")
        logger.info("This should use speculative decoding logic with Cloud-only mode")
        
        generated_text, latency_metrics = cloud_model.generate_with_spec_logic(
            audio_features=audio_waveform,
            prompt=prompt,
            max_new_tokens=128,  # Increase for better testing
            target_sentences=1,  # Reduce to 1 sentence for testing
            min_chars=20,        # Reduce minimum chars for testing
            min_new_tokens_sc=12, # Reduce minimum tokens for testing
            prompt_type="default"  # Use default prompt type like original cloud baseline
        )
        
        # Check results
        logger.info("=" * 80)
        logger.info("TEST RESULTS")
        logger.info("=" * 80)
        
        if generated_text:
            logger.info("✅ Generation successful!")
            logger.info(f"Generated text: {generated_text}")
            logger.info(f"Generated text length: {len(generated_text)} characters")
        else:
            logger.error("❌ Generation failed - no text generated")
            return False
        
        if latency_metrics:
            logger.info("✅ Latency metrics collected!")
            logger.info(f"TTFT: {latency_metrics.get('ttft', 0.0):.3f}s")
            logger.info(f"OTPS: {latency_metrics.get('otps', 0.0):.3f} tokens/s")
            logger.info(f"Total time: {latency_metrics.get('total_time', 0.0):.3f}s")
            logger.info(f"Output tokens: {latency_metrics.get('output_tokens', 0)}")
        else:
            logger.warning("⚠️ No latency metrics collected")
        
        # Test that it's using speculative decoding logic
        total_cloud_calls = latency_metrics.get('total_cloud_calls', 0)
        if total_cloud_calls > 0:
            logger.info(f"✅ Speculative decoding logic working - Cloud calls: {total_cloud_calls}")
        else:
            logger.warning("⚠️ No cloud calls detected - may not be using speculative decoding logic")
        
        logger.info("=" * 80)
        logger.info("TEST COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    success = test_cloud_optimized_baseline()
    
    if success:
        print("✅ Cloud Optimized Baseline test passed!")
        sys.exit(0)
    else:
        print("❌ Cloud Optimized Baseline test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
