#!/usr/bin/env python3
"""
Test script to verify the new detailed latency metrics
"""

import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.evaluation.metrics import EvaluationMetrics

def test_detailed_latency_metrics():
    """Test the new detailed latency metrics calculation"""
    metrics = EvaluationMetrics()
    
    # Sample detailed latency data (simulated)
    sample_latency_data = [
        {
            'ttft': 1.5,
            'itps': 100.0,
            'otps': 25.0,
            'oet': 2.0,
            'total_time': 3.5,
            'input_tokens': 150,
            'output_tokens': 50
        },
        {
            'ttft': 1.2,
            'itps': 120.0,
            'otps': 30.0,
            'oet': 1.8,
            'total_time': 3.0,
            'input_tokens': 144,
            'output_tokens': 54
        },
        {
            'ttft': 1.8,
            'itps': 80.0,
            'otps': 20.0,
            'oet': 2.5,
            'total_time': 4.3,
            'input_tokens': 144,
            'output_tokens': 50
        }
    ]
    
    # Calculate detailed metrics
    detailed_metrics = metrics.compute_detailed_latency_metrics(sample_latency_data)
    
    print("Detailed Latency Metrics Test Results:")
    print("=" * 50)
    
    # Display TTFT metrics
    print("\nTTFT (Time-to-First-Token):")
    print(f"  Mean: {detailed_metrics['ttft_mean']:.4f}s")
    print(f"  Median: {detailed_metrics['ttft_median']:.4f}s")
    print(f"  Std: {detailed_metrics['ttft_std']:.4f}s")
    print(f"  Min: {detailed_metrics['ttft_min']:.4f}s")
    print(f"  Max: {detailed_metrics['ttft_max']:.4f}s")
    print(f"  P95: {detailed_metrics['ttft_p95']:.4f}s")
    print(f"  P99: {detailed_metrics['ttft_p99']:.4f}s")
    
    # Display ITPS metrics
    print("\nITPS (Input Token Per Second):")
    print(f"  Mean: {detailed_metrics['itps_mean']:.2f} tokens/sec")
    print(f"  Median: {detailed_metrics['itps_median']:.2f} tokens/sec")
    print(f"  Std: {detailed_metrics['itps_std']:.2f} tokens/sec")
    print(f"  Min: {detailed_metrics['itps_min']:.2f} tokens/sec")
    print(f"  Max: {detailed_metrics['itps_max']:.2f} tokens/sec")
    print(f"  P95: {detailed_metrics['itps_p95']:.2f} tokens/sec")
    print(f"  P99: {detailed_metrics['itps_p99']:.2f} tokens/sec")
    
    # Display OTPS metrics
    print("\nOTPS (Output Token Per Second):")
    print(f"  Mean: {detailed_metrics['otps_mean']:.2f} tokens/sec")
    print(f"  Median: {detailed_metrics['otps_median']:.2f} tokens/sec")
    print(f"  Std: {detailed_metrics['otps_std']:.2f} tokens/sec")
    print(f"  Min: {detailed_metrics['otps_min']:.2f} tokens/sec")
    print(f"  Max: {detailed_metrics['otps_max']:.2f} tokens/sec")
    print(f"  P95: {detailed_metrics['otps_p95']:.2f} tokens/sec")
    print(f"  P99: {detailed_metrics['otps_p99']:.2f} tokens/sec")
    
    # Display OET metrics
    print("\nOET (Output Evaluation Time):")
    print(f"  Mean: {detailed_metrics['oet_mean']:.4f}s")
    print(f"  Median: {detailed_metrics['oet_median']:.4f}s")
    print(f"  Std: {detailed_metrics['oet_std']:.4f}s")
    print(f"  Min: {detailed_metrics['oet_min']:.4f}s")
    print(f"  Max: {detailed_metrics['oet_max']:.4f}s")
    print(f"  P95: {detailed_metrics['oet_p95']:.4f}s")
    print(f"  P99: {detailed_metrics['oet_p99']:.4f}s")
    
    # Display Total Time metrics
    print("\nTotal Time:")
    print(f"  Mean: {detailed_metrics['total_time_mean']:.4f}s")
    print(f"  Median: {detailed_metrics['total_time_median']:.4f}s")
    print(f"  Std: {detailed_metrics['total_time_std']:.4f}s")
    print(f"  Min: {detailed_metrics['total_time_min']:.4f}s")
    print(f"  Max: {detailed_metrics['total_time_max']:.4f}s")
    print(f"  P95: {detailed_metrics['total_time_p95']:.4f}s")
    print(f"  P99: {detailed_metrics['total_time_p99']:.4f}s")
    
    # Display Token statistics
    print("\nToken Statistics:")
    print(f"  Average Input Tokens: {detailed_metrics['avg_input_tokens']:.1f}")
    print(f"  Average Output Tokens: {detailed_metrics['avg_output_tokens']:.1f}")
    print(f"  Total Input Tokens: {detailed_metrics['total_input_tokens']}")
    print(f"  Total Output Tokens: {detailed_metrics['total_output_tokens']}")

if __name__ == "__main__":
    test_detailed_latency_metrics()
