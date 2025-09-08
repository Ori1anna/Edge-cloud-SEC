#!/usr/bin/env python3
"""
Simplified test script for data processing only
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import time
import logging
from typing import List, Dict, Any
import torch
from pathlib import Path

from src.data.audio_processor import AudioProcessor
from src.evaluation.metrics import EvaluationMetrics
from src.utils.config import load_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_processing_only(max_samples: int = 5):
    """Test data processing without loading models"""
    
    print("Testing data processing pipeline...")
    
    # Load configuration
    config = load_config("configs/default.yaml")
    print("✓ Configuration loaded")
    
    # Initialize components
    audio_processor = AudioProcessor(**config['audio'])
    metrics = EvaluationMetrics()
    print("✓ Components initialized")
    
    # Load data
    manifest_path = config['data']['train_path']
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    
    test_samples = manifest[:max_samples]
    print(f"✓ Loaded {len(test_samples)} test samples")
    
    # Test audio processing
    results = []
    for i, sample in enumerate(test_samples):
        print(f"Processing sample {i+1}/{len(test_samples)}: {sample['file_id']}")
        
        try:
            # Load audio features
            audio_path = sample['audio_path']
            print(f"  Audio path: {audio_path}")
            
            # Check if audio file exists
            if not os.path.exists(audio_path):
                print(f"  ✗ Audio file not found: {audio_path}")
                continue
            
            # Extract audio features
            start_time = time.time()
            audio_features = audio_processor.extract_features(audio_path)
            processing_time = time.time() - start_time
            
            print(f"  ✓ Audio features extracted: {audio_features.shape}, time: {processing_time:.2f}s")
            
            # Mock prediction (since we don't have models loaded)
            mock_prediction = "This is a mock prediction for testing purposes."
            
            results.append({
                'file_id': sample['file_id'],
                'dataset': sample['dataset'],
                'prediction': mock_prediction,
                'reference': sample['caption'],
                'processing_time': processing_time,
                'audio_shape': list(audio_features.shape)
            })
            
        except Exception as e:
            print(f"  ✗ Error processing sample: {e}")
            continue
    
    # Print summary
    print(f"\nData processing test completed!")
    print(f"Successfully processed: {len(results)}/{len(test_samples)} samples")
    
    if results:
        avg_time = sum(r['processing_time'] for r in results) / len(results)
        print(f"Average processing time: {avg_time:.2f}s")
        
        # Test metrics with mock data
        predictions = [r['prediction'] for r in results]
        references = [[r['reference']] for r in results]
        
        test_metrics = metrics.compute_all_metrics(predictions, references)
        print(f"Mock metrics: {test_metrics}")
    
    return results

if __name__ == "__main__":
    test_data_processing_only(max_samples=5)
