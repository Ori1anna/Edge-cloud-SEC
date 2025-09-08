#!/usr/bin/env python3
"""
Speculative decoding experiment for speech emotion captioning
Tests edge-cloud collaborative approach
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
from tqdm import tqdm

from src.models.edge_model import EdgeModel
from src.models.cloud_model import CloudModel
from src.models.speculative_decoder import SpeculativeDecoder
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


def run_speculative_experiment(config_path: str = "configs/default.yaml", 
                              max_samples: int = None):
    """Run speculative decoding experiment"""
    
    # Load configuration
    config = load_config(config_path)
    logger.info("Configuration loaded")
    
    # Initialize components
    audio_processor = AudioProcessor(**config['audio'])
    edge_model = EdgeModel(**config['models']['edge'])
    cloud_model = CloudModel(**config['models']['cloud'])
    decoder = SpeculativeDecoder(edge_model, cloud_model, config)
    metrics = EvaluationMetrics()
    
    # Load data
    manifest = load_manifest(config['data']['train_path'])
    logger.info(f"Loaded {len(manifest)} samples from manifest")
    
    # Limit samples if specified
    if max_samples:
        manifest = manifest[:max_samples]
        logger.info(f"Limited to {max_samples} samples for testing")
    
    # Run experiment
    results = []
    successful_samples = 0
    failed_samples = 0
    
    logger.info(f"Starting speculative decoding experiment on {len(manifest)} samples...")
    
    for i, sample in enumerate(tqdm(manifest, desc="Processing samples")):
        try:
            # Load audio features
            audio_path = sample['audio_path']
            audio_features = audio_processor.extract_features(audio_path)
            
            # Speculative decoding
            start_time = time.time()
            spec_tokens = decoder.decode(audio_features)
            spec_latency = time.time() - start_time
            spec_text = edge_model.tokenizer.decode(spec_tokens)
            
            results.append({
                'file_id': sample['file_id'],
                'dataset': sample['dataset'],
                'prediction': spec_text,
                'reference': sample['caption'],
                'latency': spec_latency,
                'tokens_generated': len(spec_tokens)
            })
            
            successful_samples += 1
            
        except Exception as e:
            logger.error(f"Error processing sample {sample['file_id']}: {e}")
            failed_samples += 1
            continue
    
    # Compute metrics
    if results:
        predictions = [r['prediction'] for r in results]
        references = [[r['reference']] for r in results]
        latencies = [r['latency'] for r in results]
        
        final_metrics = metrics.compute_all_metrics(
            predictions, references, latencies=latencies
        )
        
        logger.info(f"\nSPECULATIVE DECODING Final Results:")
        for metric, value in final_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
    
    # Save results
    output_dir = Path("experiments/results")
    output_dir.mkdir(exist_ok=True)
    
    final_data = {
        'experiment_type': 'speculative_decoding',
        'experiment_info': {
            'total_samples': successful_samples + failed_samples,
            'successful_samples': successful_samples,
            'failed_samples': failed_samples,
            'success_rate': successful_samples / (successful_samples + failed_samples),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'metrics': final_metrics if results else {},
        'detailed_results': results
    }
    
    with open(output_dir / "speculative_results.json", 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Speculative decoding results saved to {output_dir / 'speculative_results.json'}")
    logger.info(f"Experiment completed. Success: {successful_samples}, Failed: {failed_samples}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run speculative decoding experiment")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to process")
    
    args = parser.parse_args()
    
    run_speculative_experiment(
        config_path=args.config,
        max_samples=args.max_samples
    )
