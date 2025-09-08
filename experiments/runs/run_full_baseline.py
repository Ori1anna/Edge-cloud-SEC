#!/usr/bin/env python3
"""
Full baseline experiment for speech emotion captioning
Runs experiments on the complete dataset
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


def run_full_baseline_experiment(config_path: str = "configs/default.yaml", 
                                max_samples: int = None,
                                save_interval: int = 100):
    """Run full baseline experiment on complete dataset"""
    
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
    
    # Run experiments
    results = {
        'edge_only': [],
        'cloud_only': [],
        'speculative': []
    }
    
    # Progress tracking
    successful_samples = 0
    failed_samples = 0
    
    logger.info(f"Starting experiment on {len(manifest)} samples...")
    
    for i, sample in enumerate(tqdm(manifest, desc="Processing samples")):
        try:
            # Load audio features
            audio_path = sample['audio_path']
            audio_features = audio_processor.extract_features(audio_path)
            
            # Edge-only baseline
            start_time = time.time()
            edge_tokens, edge_log_probs = edge_model.generate_draft(audio_features)
            edge_latency = time.time() - start_time
            edge_text = edge_model.tokenizer.decode(edge_tokens)
            
            results['edge_only'].append({
                'file_id': sample['file_id'],
                'prediction': edge_text,
                'reference': sample['caption'],
                'latency': edge_latency,
                'dataset': sample['dataset']
            })
            
            # Cloud-only baseline
            start_time = time.time()
            cloud_tokens, verification_length = cloud_model.verify_tokens(edge_tokens, audio_features)
            cloud_latency = time.time() - start_time
            cloud_text = cloud_model.tokenizer.decode(cloud_tokens)
            
            results['cloud_only'].append({
                'file_id': sample['file_id'],
                'prediction': cloud_text,
                'reference': sample['caption'],
                'latency': cloud_latency,
                'dataset': sample['dataset']
            })
            
            # Speculative decoding
            start_time = time.time()
            spec_tokens = decoder.decode(audio_features)
            spec_latency = time.time() - start_time
            spec_text = edge_model.tokenizer.decode(spec_tokens)
            
            results['speculative'].append({
                'file_id': sample['file_id'],
                'prediction': spec_text,
                'reference': sample['caption'],
                'latency': spec_latency,
                'dataset': sample['dataset']
            })
            
            successful_samples += 1
            
            # Save intermediate results
            if (i + 1) % save_interval == 0:
                _save_intermediate_results(results, i + 1, successful_samples, failed_samples)
                
        except Exception as e:
            logger.error(f"Error processing sample {sample['file_id']}: {e}")
            failed_samples += 1
            continue
    
    # Final metrics computation
    logger.info("Computing final metrics...")
    final_metrics = {}
    
    for method in results:
        if results[method]:
            predictions = [r['prediction'] for r in results[method]]
            references = [[r['reference']] for r in results[method]]
            latencies = [r['latency'] for r in results[method]]
            
            method_metrics = metrics.compute_all_metrics(
                predictions, references, latencies=latencies
            )
            
            final_metrics[method] = method_metrics
            
            logger.info(f"\n{method.upper()} Final Results:")
            for metric, value in method_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
    
    # Save final results
    _save_final_results(results, final_metrics, successful_samples, failed_samples)
    
    logger.info(f"Experiment completed. Success: {successful_samples}, Failed: {failed_samples}")


def _save_intermediate_results(results: Dict, processed_count: int, 
                             successful_samples: int, failed_samples: int):
    """Save intermediate results during experiment"""
    output_dir = Path("experiments/results")
    output_dir.mkdir(exist_ok=True)
    
    intermediate_data = {
        'processed_count': processed_count,
        'successful_samples': successful_samples,
        'failed_samples': failed_samples,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': results
    }
    
    with open(output_dir / f"intermediate_results_{processed_count}.json", 'w', encoding='utf-8') as f:
        json.dump(intermediate_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved intermediate results for {processed_count} samples")


def _save_final_results(results: Dict, metrics: Dict, 
                       successful_samples: int, failed_samples: int):
    """Save final experiment results"""
    output_dir = Path("experiments/results")
    output_dir.mkdir(exist_ok=True)
    
    final_data = {
        'experiment_info': {
            'total_samples': successful_samples + failed_samples,
            'successful_samples': successful_samples,
            'failed_samples': failed_samples,
            'success_rate': successful_samples / (successful_samples + failed_samples),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'metrics': metrics,
        'detailed_results': results
    }
    
    with open(output_dir / "full_baseline_results.json", 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Final results saved to {output_dir / 'full_baseline_results.json'}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run full baseline experiment")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to process")
    parser.add_argument("--save-interval", type=int, default=100, help="Save interval for intermediate results")
    
    args = parser.parse_args()
    
    run_full_baseline_experiment(
        config_path=args.config,
        max_samples=args.max_samples,
        save_interval=args.save_interval
    )
