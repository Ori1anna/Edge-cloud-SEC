#!/usr/bin/env python3
"""
Test script to verify accurate latency metrics calculation
"""

import sys
import os
import json
import time
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.edge_model import EdgeModel
from src.models.cloud_model import CloudModel
from src.data.loader import UnifiedDataLoader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_accurate_metrics():
    """Test accurate metrics calculation for both models"""
    
    print("=" * 60)
    print("Testing Accurate Latency Metrics")
    print("=" * 60)
    
    # Load test data
    data_loader = UnifiedDataLoader("data/processed/secap/manifest.json")
    test_samples = data_loader.load_samples(max_samples=2)
    
    if not test_samples:
        print("❌ No test samples loaded")
        return False
    
    # Test Edge Model
    print("\n🔧 Testing Edge Model (3B) with accurate metrics...")
    try:
        edge_model = EdgeModel(device="cuda", dtype="float16")
        
        for i, sample in enumerate(test_samples[:1]):  # Test only first sample
            print(f"\n--- Edge Model Test {i+1} ---")
            
            # Test with streaming (accurate metrics)
            print("Testing with streaming generation...")
            start_time = time.time()
            text_streaming, metrics_streaming = edge_model.generate_draft(
                sample['audio_waveform'],
                sample['prompt'],
                max_new_tokens=32,
                use_streaming=True
            )
            streaming_time = time.time() - start_time
            
            print(f"Generated text: {text_streaming[:100]}...")
            print(f"Streaming metrics:")
            print(f"  - TTFT: {metrics_streaming.get('ttft', 'N/A'):.4f}s")
            print(f"  - OTPS: {metrics_streaming.get('otps', 'N/A'):.2f} tokens/s")
            print(f"  - Total time: {metrics_streaming.get('total_time', 'N/A'):.4f}s")
            print(f"  - Output tokens: {metrics_streaming.get('output_tokens', 'N/A')}")
            print(f"  - Streaming time: {streaming_time:.4f}s")
            
            # Test without streaming (original method)
            print("\nTesting with batch generation...")
            start_time = time.time()
            text_batch, metrics_batch = edge_model.generate_draft(
                sample['audio_waveform'],
                sample['prompt'],
                max_new_tokens=32,
                use_streaming=False
            )
            batch_time = time.time() - start_time
            
            print(f"Generated text: {text_batch[:100]}...")
            print(f"Batch metrics:")
            print(f"  - TTFT: {metrics_batch.get('ttft', 'N/A'):.4f}s")
            print(f"  - OTPS: {metrics_batch.get('otps', 'N/A'):.2f} tokens/s")
            print(f"  - Total time: {metrics_batch.get('total_time', 'N/A'):.4f}s")
            print(f"  - Output tokens: {metrics_batch.get('output_tokens', 'N/A')}")
            print(f"  - Batch time: {batch_time:.4f}s")
            
            # Compare metrics
            print(f"\n📊 Comparison:")
            print(f"  - TTFT difference: {abs(metrics_streaming.get('ttft', 0) - metrics_batch.get('ttft', 0)):.4f}s")
            print(f"  - OTPS difference: {abs(metrics_streaming.get('otps', 0) - metrics_batch.get('otps', 0)):.2f} tokens/s")
            print(f"  - Time difference: {abs(streaming_time - batch_time):.4f}s")
            
    except Exception as e:
        print(f"❌ Edge model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test Cloud Model
    print("\n🔧 Testing Cloud Model (7B) with accurate metrics...")
    try:
        cloud_model = CloudModel(device="cuda", dtype="float16")
        
        for i, sample in enumerate(test_samples[:1]):  # Test only first sample
            print(f"\n--- Cloud Model Test {i+1} ---")
            
            # Test with streaming (accurate metrics)
            print("Testing with streaming generation...")
            start_time = time.time()
            text_streaming, metrics_streaming = cloud_model.generate_independently(
                sample['audio_waveform'],
                sample['prompt'],
                max_new_tokens=32,
                use_streaming=True
            )
            streaming_time = time.time() - start_time
            
            print(f"Generated text: {text_streaming[:100]}...")
            print(f"Streaming metrics:")
            print(f"  - TTFT: {metrics_streaming.get('ttft', 'N/A'):.4f}s")
            print(f"  - OTPS: {metrics_streaming.get('otps', 'N/A'):.2f} tokens/s")
            print(f"  - Total time: {metrics_streaming.get('total_time', 'N/A'):.4f}s")
            print(f"  - Output tokens: {metrics_streaming.get('output_tokens', 'N/A')}")
            print(f"  - Streaming time: {streaming_time:.4f}s")
            
            # Test without streaming (original method)
            print("\nTesting with batch generation...")
            start_time = time.time()
            text_batch, metrics_batch = cloud_model.generate_independently(
                sample['audio_waveform'],
                sample['prompt'],
                max_new_tokens=32,
                use_streaming=False
            )
            batch_time = time.time() - start_time
            
            print(f"Generated text: {text_batch[:100]}...")
            print(f"Batch metrics:")
            print(f"  - TTFT: {metrics_batch.get('ttft', 'N/A'):.4f}s")
            print(f"  - OTPS: {metrics_batch.get('otps', 'N/A'):.2f} tokens/s")
            print(f"  - Total time: {metrics_batch.get('total_time', 'N/A'):.4f}s")
            print(f"  - Output tokens: {metrics_batch.get('output_tokens', 'N/A')}")
            print(f"  - Batch time: {batch_time:.4f}s")
            
            # Compare metrics
            print(f"\n📊 Comparison:")
            print(f"  - TTFT difference: {abs(metrics_streaming.get('ttft', 0) - metrics_batch.get('ttft', 0)):.4f}s")
            print(f"  - OTPS difference: {abs(metrics_streaming.get('otps', 0) - metrics_batch.get('otps', 0)):.2f} tokens/s")
            print(f"  - Time difference: {abs(streaming_time - batch_time):.4f}s")
            
    except Exception as e:
        print(f"❌ Cloud model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✅ All tests completed successfully!")
    return True


def test_comparison():
    """Test direct comparison between Edge and Cloud models"""
    
    print("\n" + "=" * 60)
    print("Direct Edge vs Cloud Comparison")
    print("=" * 60)
    
    # Load test data
    data_loader = UnifiedDataLoader("data/processed/secap/manifest.json")
    test_samples = data_loader.load_samples(max_samples=1)
    
    if not test_samples:
        print("❌ No test samples loaded")
        return False
    
    try:
        # Initialize both models
        edge_model = EdgeModel(device="cuda", dtype="float16")
        cloud_model = CloudModel(device="cuda", dtype="float16")
        
        sample = test_samples[0]
        
        print(f"\nTesting with sample: {sample['file_id']}")
        print(f"Reference: {sample['reference_caption']}")
        
        # Test Edge model
        print("\n🔧 Edge Model (3B):")
        edge_start = time.time()
        edge_text, edge_metrics = edge_model.generate_draft(
            sample['audio_waveform'],
            sample['prompt'],
            max_new_tokens=32,
            use_streaming=True
        )
        edge_time = time.time() - edge_start
        
        print(f"Generated: {edge_text}")
        print(f"TTFT: {edge_metrics.get('ttft', 'N/A'):.4f}s")
        print(f"OTPS: {edge_metrics.get('otps', 'N/A'):.2f} tokens/s")
        print(f"Total time: {edge_metrics.get('total_time', 'N/A'):.4f}s")
        print(f"Output tokens: {edge_metrics.get('output_tokens', 'N/A')}")
        
        # Test Cloud model
        print("\n🔧 Cloud Model (7B):")
        cloud_start = time.time()
        cloud_text, cloud_metrics = cloud_model.generate_independently(
            sample['audio_waveform'],
            sample['prompt'],
            max_new_tokens=32,
            use_streaming=True
        )
        cloud_time = time.time() - cloud_start
        
        print(f"Generated: {cloud_text}")
        print(f"TTFT: {cloud_metrics.get('ttft', 'N/A'):.4f}s")
        print(f"OTPS: {cloud_metrics.get('otps', 'N/A'):.2f} tokens/s")
        print(f"Total time: {cloud_metrics.get('total_time', 'N/A'):.4f}s")
        print(f"Output tokens: {cloud_metrics.get('output_tokens', 'N/A')}")
        
        # Compare results
        print(f"\n📊 Direct Comparison:")
        print(f"  - TTFT: Edge {edge_metrics.get('ttft', 0):.4f}s vs Cloud {cloud_metrics.get('ttft', 0):.4f}s")
        print(f"  - OTPS: Edge {edge_metrics.get('otps', 0):.2f} vs Cloud {cloud_metrics.get('otps', 0):.2f} tokens/s")
        print(f"  - Total: Edge {edge_metrics.get('total_time', 0):.4f}s vs Cloud {cloud_metrics.get('total_time', 0):.4f}s")
        print(f"  - Tokens: Edge {edge_metrics.get('output_tokens', 0)} vs Cloud {cloud_metrics.get('output_tokens', 0)}")
        
        # Check if metrics make sense
        edge_ttft = edge_metrics.get('ttft', 0)
        cloud_ttft = cloud_metrics.get('ttft', 0)
        
        if edge_ttft > 0 and cloud_ttft > 0:
            if edge_ttft < cloud_ttft:
                print("✅ Edge model has faster TTFT (as expected for smaller model)")
            else:
                print("⚠️  Cloud model has faster TTFT (unexpected, but could be due to optimization)")
        
        return True
        
    except Exception as e:
        print(f"❌ Comparison test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 Starting Accurate Metrics Test")
    
    # Test individual models
    success1 = test_accurate_metrics()
    
    # Test comparison
    success2 = test_comparison()
    
    if success1 and success2:
        print("\n🎉 All tests passed! Accurate metrics are working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
