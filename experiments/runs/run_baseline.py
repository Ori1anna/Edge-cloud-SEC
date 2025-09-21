#!/usr/bin/env python3
"""
Baseline experiment for speech emotion captioning
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import logging
from typing import List, Dict, Any
import torch
from pathlib import Path

from src.models.edge_model import EdgeModel
from src.models.cloud_model import CloudModel
from src.data.audio_processor import AudioProcessor
from src.evaluation.metrics import EvaluationMetrics
from src.utils.config import load_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_manifest(manifest_path: str) -> List[Dict[str, Any]]:
    """Load unified manifest"""
    with open(manifest_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_baseline_experiment(config_path: str = "configs/default.yaml"):
    """Run baseline experiment"""
    
    # Load configuration
    config = load_config(config_path)
    logger.info("Configuration loaded")
    
    # Initialize components
    audio_processor = AudioProcessor(**config['audio'])
    edge_model = EdgeModel(**config['models']['edge'])
    cloud_model = CloudModel(**config['models']['cloud'])
    metrics = EvaluationMetrics()
    
    # Load data
    manifest = load_manifest(config['data']['train_path'])
    logger.info(f"Loaded {len(manifest)} samples from manifest")
    
    # Take a subset for testing
    test_samples = manifest[:10]  # Test with first 10 samples
    
    # Run experiments
    results = {
        'edge_only': [],
        'cloud_only': [],
    }
    
    for i, sample in enumerate(test_samples):
        logger.info(f"Processing sample {i+1}/{len(test_samples)}: {sample['file_id']}")
        
        try:
            # Load audio features
            audio_path = sample['audio_path']
            audio_features = audio_processor.extract_features(audio_path)
            
            # Edge-only baseline
            start_time = time.time()
            edge_tokens = edge_model.generate_draft(audio_features)[0]
            edge_latency = time.time() - start_time
            edge_text = edge_model.tokenizer.decode(edge_tokens)
            
            results['edge_only'].append({
                'file_id': sample['file_id'],
                'prediction': edge_text,
                'reference': sample['caption'],
                'latency': edge_latency
            })
            
            # Cloud-only baseline
            start_time = time.time()
            # Use cloud model to generate independently (not verifying edge tokens)
            cloud_tokens, _ = cloud_model.generate_independently(audio_features)
            cloud_latency = time.time() - start_time
            cloud_text = cloud_model.tokenizer.decode(cloud_tokens)
            
            results['cloud_only'].append({
                'file_id': sample['file_id'],
                'prediction': cloud_text,
                'reference': sample['caption'],
                'latency': cloud_latency
            })
            
            
        except Exception as e:
            logger.error(f"Error processing sample {sample['file_id']}: {e}")
            continue
    
    # Compute metrics
    for method in results:
        if results[method]:
            predictions = [r['prediction'] for r in results[method]]
            references = [[r['reference']] for r in results[method]]
            latencies = [r['latency'] for r in results[method]]
            
            method_metrics = metrics.compute_all_metrics(
                predictions, references, latencies=latencies
            )
            
            logger.info(f"\n{method.upper()} Results:")
            for metric, value in method_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
    
    # Save results
    output_dir = Path("experiments/results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "baseline_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_dir / 'baseline_results.json'}")


if __name__ == "__main__":
    run_baseline_experiment()
