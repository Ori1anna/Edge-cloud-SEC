#!/usr/bin/env python3
"""
Test script to verify the simplified latency metrics calculation
"""

import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.evaluation.metrics import EvaluationMetrics

def test_simplified_metrics():
    """Test the simplified latency metrics calculation"""
    metrics = EvaluationMetrics()
    
    # Sample detailed latency data
    sample_latency_data = [
        {
            'ttft': 2.5,
            'itps': 100.0,
            'otps': 25.0,
            'oet': 2.5,
            'total_time': 2.6,
            'input_tokens': 150,
            'output_tokens': 50
        },
        {
            'ttft': 3.0,
            'itps': 120.0,
            'otps': 30.0,
            'oet': 3.0,
            'total_time': 3.1,
            'input_tokens': 144,
            'output_tokens': 54
        },
        {
            'ttft': 4.0,
            'itps': 80.0,
            'otps': 20.0,
            'oet': 4.0,
            'total_time': 4.1,
            'input_tokens': 144,
            'output_tokens': 50
        }
    ]
    
    # Calculate simplified metrics
    simplified_metrics = metrics.compute_detailed_latency_metrics(sample_latency_data)
    
    print("Simplified Latency Metrics Test Results:")
    print("=" * 50)
    
    # Display TTFT metrics
    print("\nTTFT (Time-to-First-Token):")
    print(f"  Mean: {simplified_metrics['ttft_mean']:.4f}s")
    print(f"  Min: {simplified_metrics['ttft_min']:.4f}s")
    print(f"  Max: {simplified_metrics['ttft_max']:.4f}s")
    
    # Display ITPS metrics
    print("\nITPS (Input Token Per Second):")
    print(f"  Mean: {simplified_metrics['itps_mean']:.2f} tokens/sec")
    print(f"  Min: {simplified_metrics['itps_min']:.2f} tokens/sec")
    print(f"  Max: {simplified_metrics['itps_max']:.2f} tokens/sec")
    
    # Display OTPS metrics
    print("\nOTPS (Output Token Per Second):")
    print(f"  Mean: {simplified_metrics['otps_mean']:.2f} tokens/sec")
    print(f"  Min: {simplified_metrics['otps_min']:.2f} tokens/sec")
    print(f"  Max: {simplified_metrics['otps_max']:.2f} tokens/sec")
    
    # Display OET metrics
    print("\nOET (Output Evaluation Time):")
    print(f"  Mean: {simplified_metrics['oet_mean']:.4f}s")
    print(f"  Min: {simplified_metrics['oet_min']:.4f}s")
    print(f"  Max: {simplified_metrics['oet_max']:.4f}s")
    
    # Display Total Time metrics
    print("\nTotal Time:")
    print(f"  Mean: {simplified_metrics['total_time_mean']:.4f}s")
    print(f"  Min: {simplified_metrics['total_time_min']:.4f}s")
    print(f"  Max: {simplified_metrics['total_time_max']:.4f}s")
    
    # Display Token statistics
    print("\nToken Statistics:")
    print(f"  Average Input Tokens: {simplified_metrics['avg_input_tokens']:.1f}")
    print(f"  Average Output Tokens: {simplified_metrics['avg_output_tokens']:.1f}")
    print(f"  Total Input Tokens: {simplified_metrics['total_input_tokens']}")
    print(f"  Total Output Tokens: {simplified_metrics['total_output_tokens']}")
    
    # Verify that only mean, min, max are present
    print("\nVerification - Only mean, min, max keys should exist:")
    for key in simplified_metrics.keys():
        if 'median' in key or 'std' in key or 'p95' in key or 'p99' in key:
            print(f"  ❌ Unexpected key found: {key}")
        else:
            print(f"  ✅ Valid key: {key}")

if __name__ == "__main__":
    test_simplified_metrics()
