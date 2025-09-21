#!/usr/bin/env python3
"""
Run baseline experiments with accurate latency metrics
"""

import sys
import os
import json
import time
import logging
from pathlib import Path
from argparse import ArgumentParser

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from models.edge_model import EdgeModel
from models.cloud_model import CloudModel
from data.loader import UnifiedDataLoader
from evaluation.metrics import calculate_bleu_cider, calculate_latency_metrics

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_accurate_edge_baseline(dataset_path: str, output_name: str, 
                              prompt_type: str = "default", language: str = "chinese",
                              max_samples: int = 10, use_streaming: bool = True):
    """Run Edge baseline with accurate metrics"""
    
    print("=" * 60)
    print("Running Edge Baseline with Accurate Metrics")
    print("=" * 60)
    
    # Load data
    data_loader = UnifiedDataLoader(dataset_path)
    samples = data_loader.load_samples(max_samples=max_samples)
    
    if not samples:
        print("❌ No samples loaded")
        return
    
    print(f"Loaded {len(samples)} samples")
    
    # Initialize Edge model
    print("Initializing Edge model...")
    edge_model = EdgeModel(device="cuda", dtype="float16")
    
    # Run experiments
    results = []
    total_start_time = time.time()
    
    for i, sample in enumerate(samples):
        print(f"\n--- Processing sample {i+1}/{len(samples)} ---")
        print(f"File ID: {sample['file_id']}")
        print(f"Reference: {sample['reference_caption']}")
        
        try:
            # Generate with Edge model
            sample_start_time = time.time()
            generated_text, latency_metrics = edge_model.generate_draft(
                sample['audio_waveform'],
                sample['prompt'],
                max_new_tokens=32,
                use_streaming=use_streaming
            )
            sample_time = time.time() - sample_start_time
            
            print(f"Generated: {generated_text}")
            print(f"TTFT: {latency_metrics.get('ttft', 'N/A'):.4f}s")
            print(f"OTPS: {latency_metrics.get('otps', 'N/A'):.2f} tokens/s")
            print(f"Total time: {latency_metrics.get('total_time', 'N/A'):.4f}s")
            
            # Calculate BLEU and CIDEr scores
            bleu_score = calculate_bleu_cider([generated_text], [sample['reference_caption']])['bleu']
            cider_score = calculate_bleu_cider([generated_text], [sample['reference_caption']])['cider']
            
            # Store result
            result = {
                'file_id': sample['file_id'],
                'dataset': sample.get('dataset', 'unknown'),
                'reference_caption': sample['reference_caption'],
                'generated_text': generated_text,
                'bleu_score': bleu_score,
                'cider_score': cider_score,
                'caption_type': sample.get('caption_type', 'original'),
                'language': language,
                'prompt_type': prompt_type,
                'latency_metrics': latency_metrics,
                'sample_processing_time': sample_time
            }
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing sample {i+1}: {e}")
            continue
    
    total_time = time.time() - total_start_time
    
    # Calculate overall metrics
    if results:
        bleu_scores = [r['bleu_score'] for r in results]
        cider_scores = [r['cider_score'] for r in results]
        
        # Calculate latency metrics
        latency_metrics_list = [r['latency_metrics'] for r in results if r['latency_metrics']]
        
        overall_metrics = {
            'avg_bleu': sum(bleu_scores) / len(bleu_scores),
            'avg_cider': sum(cider_scores) / len(cider_scores),
            'latency_metrics': calculate_latency_metrics(latency_metrics_list),
            'total_samples': len(results),
            'total_processing_time': total_time
        }
        
        # Create output
        output_data = {
            'experiment_config': {
                'dataset_type': 'unified',
                'dataset_path': dataset_path,
                'caption_type': 'original',
                'language': language,
                'prompt_type': prompt_type,
                'max_samples': max_samples,
                'total_samples': len(results),
                'use_streaming': use_streaming
            },
            'metrics': overall_metrics,
            'detailed_results': results
        }
        
        # Save results
        output_path = f"experiments/results/{output_name}_edge_accurate.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Results saved to {output_path}")
        print(f"Average BLEU: {overall_metrics['avg_bleu']:.4f}")
        print(f"Average CIDEr: {overall_metrics['avg_cider']:.4f}")
        print(f"Average TTFT: {overall_metrics['latency_metrics']['ttft_mean']:.4f}s")
        print(f"Average OTPS: {overall_metrics['latency_metrics']['otps_mean']:.2f} tokens/s")
        print(f"Total processing time: {total_time:.2f}s")
        
    else:
        print("❌ No successful results")


def run_accurate_cloud_baseline(dataset_path: str, output_name: str,
                               prompt_type: str = "default", language: str = "chinese",
                               max_samples: int = 10, use_streaming: bool = True):
    """Run Cloud baseline with accurate metrics"""
    
    print("=" * 60)
    print("Running Cloud Baseline with Accurate Metrics")
    print("=" * 60)
    
    # Load data
    data_loader = UnifiedDataLoader(dataset_path)
    samples = data_loader.load_samples(max_samples=max_samples)
    
    if not samples:
        print("❌ No samples loaded")
        return
    
    print(f"Loaded {len(samples)} samples")
    
    # Initialize Cloud model
    print("Initializing Cloud model...")
    cloud_model = CloudModel(device="cuda", dtype="float16")
    
    # Run experiments
    results = []
    total_start_time = time.time()
    
    for i, sample in enumerate(samples):
        print(f"\n--- Processing sample {i+1}/{len(samples)} ---")
        print(f"File ID: {sample['file_id']}")
        print(f"Reference: {sample['reference_caption']}")
        
        try:
            # Generate with Cloud model
            sample_start_time = time.time()
            generated_text, latency_metrics = cloud_model.generate_independently(
                sample['audio_waveform'],
                sample['prompt'],
                max_new_tokens=32,
                use_streaming=use_streaming
            )
            sample_time = time.time() - sample_start_time
            
            print(f"Generated: {generated_text}")
            print(f"TTFT: {latency_metrics.get('ttft', 'N/A'):.4f}s")
            print(f"OTPS: {latency_metrics.get('otps', 'N/A'):.2f} tokens/s")
            print(f"Total time: {latency_metrics.get('total_time', 'N/A'):.4f}s")
            
            # Calculate BLEU and CIDEr scores
            bleu_score = calculate_bleu_cider([generated_text], [sample['reference_caption']])['bleu']
            cider_score = calculate_bleu_cider([generated_text], [sample['reference_caption']])['cider']
            
            # Store result
            result = {
                'file_id': sample['file_id'],
                'dataset': sample.get('dataset', 'unknown'),
                'reference_caption': sample['reference_caption'],
                'generated_text': generated_text,
                'bleu_score': bleu_score,
                'cider_score': cider_score,
                'caption_type': sample.get('caption_type', 'original'),
                'language': language,
                'prompt_type': prompt_type,
                'latency_metrics': latency_metrics,
                'sample_processing_time': sample_time
            }
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing sample {i+1}: {e}")
            continue
    
    total_time = time.time() - total_start_time
    
    # Calculate overall metrics
    if results:
        bleu_scores = [r['bleu_score'] for r in results]
        cider_scores = [r['cider_score'] for r in results]
        
        # Calculate latency metrics
        latency_metrics_list = [r['latency_metrics'] for r in results if r['latency_metrics']]
        
        overall_metrics = {
            'avg_bleu': sum(bleu_scores) / len(bleu_scores),
            'avg_cider': sum(cider_scores) / len(cider_scores),
            'latency_metrics': calculate_latency_metrics(latency_metrics_list),
            'total_samples': len(results),
            'total_processing_time': total_time
        }
        
        # Create output
        output_data = {
            'experiment_config': {
                'dataset_type': 'unified',
                'dataset_path': dataset_path,
                'caption_type': 'original',
                'language': language,
                'prompt_type': prompt_type,
                'max_samples': max_samples,
                'total_samples': len(results),
                'use_streaming': use_streaming
            },
            'metrics': overall_metrics,
            'detailed_results': results
        }
        
        # Save results
        output_path = f"experiments/results/{output_name}_cloud_accurate.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Results saved to {output_path}")
        print(f"Average BLEU: {overall_metrics['avg_bleu']:.4f}")
        print(f"Average CIDEr: {overall_metrics['avg_cider']:.4f}")
        print(f"Average TTFT: {overall_metrics['latency_metrics']['ttft_mean']:.4f}s")
        print(f"Average OTPS: {overall_metrics['latency_metrics']['otps_mean']:.2f} tokens/s")
        print(f"Total processing time: {total_time:.2f}s")
        
    else:
        print("❌ No successful results")


def main():
    parser = ArgumentParser(description="Run baseline experiments with accurate metrics")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset manifest")
    parser.add_argument("--output_name", type=str, required=True, help="Output file name")
    parser.add_argument("--model_type", type=str, choices=["edge", "cloud", "both"], default="both", help="Model type to test")
    parser.add_argument("--prompt_type", type=str, default="default", help="Prompt type")
    parser.add_argument("--language", type=str, default="chinese", help="Language")
    parser.add_argument("--max_samples", type=int, default=10, help="Maximum number of samples")
    parser.add_argument("--use_streaming", action="store_true", default=True, help="Use streaming generation")
    parser.add_argument("--no_streaming", action="store_true", help="Disable streaming generation")
    
    args = parser.parse_args()
    
    # Override streaming setting if no_streaming is specified
    if args.no_streaming:
        args.use_streaming = False
    
    print(f"Configuration:")
    print(f"  - Dataset: {args.dataset_path}")
    print(f"  - Output: {args.output_name}")
    print(f"  - Model: {args.model_type}")
    print(f"  - Prompt: {args.prompt_type}")
    print(f"  - Language: {args.language}")
    print(f"  - Max samples: {args.max_samples}")
    print(f"  - Streaming: {args.use_streaming}")
    
    if args.model_type in ["edge", "both"]:
        run_accurate_edge_baseline(
            args.dataset_path, args.output_name, args.prompt_type, 
            args.language, args.max_samples, args.use_streaming
        )
    
    if args.model_type in ["cloud", "both"]:
        run_accurate_cloud_baseline(
            args.dataset_path, args.output_name, args.prompt_type,
            args.language, args.max_samples, args.use_streaming
        )


if __name__ == "__main__":
    main()


