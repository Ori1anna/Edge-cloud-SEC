#!/usr/bin/env python3
"""
S2 Strategy Implementation: Fusion of Audio-based Emotion Descriptions with Text Transcripts
Using Qwen-3-7B-Instruct model to integrate audio emotion descriptions with original transcriptions
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "Qwen/Qwen3-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"正在使用的设备: {DEVICE}")

# S2 Fusion Prompt Template (adapted from the paper)
S2_PROMPT_TEMPLATE = """Please act as an expert in the field of emotions. We provide acoustic clues that may be related to the character's emotional state, along with the original subtitle of the video. Please analyze which parts can infer the emotional state and explain the reasons. During the analysis, please integrate the textual and audio clues.

Acoustic Clues:
{audio_description}

Original Subtitle:
{subtitle}

Integrated Analysis:"""

def load_audio_descriptions(description_file: str) -> Dict[str, str]:
    """Load audio emotion descriptions from experiment results file"""
    logger.info(f"Loading audio descriptions from {description_file}")
    
    try:
        with open(description_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract generated descriptions from detailed_results
        descriptions = {}
        if 'detailed_results' in data:
            for item in data['detailed_results']:
                file_id = item.get('file_id')
                generated_text = item.get('generated_text', '')
                if file_id and generated_text:
                    descriptions[file_id] = generated_text
        
        logger.info(f"Loaded {len(descriptions)} audio descriptions")
        return descriptions
        
    except FileNotFoundError:
        logger.error(f"Audio description file not found: {description_file}")
        return {}
    except Exception as e:
        logger.error(f"Error loading audio descriptions: {e}")
        return {}

def load_transcriptions(transcription_file: str, language: str = "english") -> Dict[str, str]:
    """Load transcriptions from manifest file"""
    logger.info(f"Loading transcriptions from {transcription_file}")
    
    try:
        with open(transcription_file, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        transcriptions = {}
        transcription_field = f"{language}_transcription"
        
        for item in manifest:
            file_id = item.get('file_id')
            transcription = item.get(transcription_field, '')
            if file_id and transcription:
                transcriptions[file_id] = transcription
        
        logger.info(f"Loaded {len(transcriptions)} transcriptions")
        return transcriptions
        
    except FileNotFoundError:
        logger.error(f"Transcription file not found: {transcription_file}")
        return {}
    except Exception as e:
        logger.error(f"Error loading transcriptions: {e}")
        return {}

def merge_data(descriptions: Dict[str, str], transcriptions: Dict[str, str]) -> List[Dict[str, Any]]:
    """Merge audio descriptions and transcriptions by file_id"""
    logger.info("Merging audio descriptions with transcriptions")
    
    merged_data = []
    common_ids = set(descriptions.keys()) & set(transcriptions.keys())
    
    for file_id in common_ids:
        merged_data.append({
            "file_id": file_id,
            "audio_description": descriptions[file_id],
            "transcription": transcriptions[file_id]
        })
    
    logger.info(f"Successfully merged {len(merged_data)} samples")
    return merged_data

def generate_fusion_descriptions(data: List[Dict[str, Any]], model, tokenizer) -> List[Dict[str, Any]]:
    """Generate integrated descriptions using Qwen model"""
    logger.info("Generating fusion descriptions...")
    
    results = []
    
    for item in tqdm(data, desc="Generating fusion descriptions"):
        try:
            # Format the prompt
            full_prompt = S2_PROMPT_TEMPLATE.format(
                audio_description=item['audio_description'],
                subtitle=item['transcription']
            )
            
            # Format for instruction-tuned model
            messages = [
                {"role": "user", "content": full_prompt}
            ]
            
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize and generate
            model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode the generated text
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            fusion_description = response[0].strip()
            
            results.append({
                "file_id": item['file_id'],
                "audio_description": item['audio_description'],
                "original_transcription": item['transcription'],
                "fusion_description": fusion_description,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
        except Exception as e:
            logger.error(f"Error generating description for {item['file_id']}: {e}")
            results.append({
                "file_id": item['file_id'],
                "audio_description": item['audio_description'],
                "original_transcription": item['transcription'],
                "fusion_description": f"Error: {str(e)}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
    
    return results

def run_s2_fusion_experiment(
    audio_description_file: str,
    transcription_file: str,
    output_file: str,
    language: str = "english",
    max_samples: Optional[int] = None,
    verbose: bool = False
):
    """
    Run S2 fusion experiment
    
    Args:
        audio_description_file: Path to audio description results file
        transcription_file: Path to manifest file with transcriptions
        output_file: Path for output file
        language: Language for transcriptions ("english" or "chinese")
        max_samples: Maximum number of samples to process
        verbose: Whether to print detailed information
    """
    
    logger.info(f"Starting S2 fusion experiment")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Language: {language}")
    
    # Load model and tokenizer
    print(f"正在加载模型: {MODEL_NAME}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print("模型加载完成。")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    
    # Load data
    audio_descriptions = load_audio_descriptions(audio_description_file)
    transcriptions = load_transcriptions(transcription_file, language)
    
    if not audio_descriptions or not transcriptions:
        logger.error("Failed to load required data")
        return
    
    # Merge data
    merged_data = merge_data(audio_descriptions, transcriptions)
    
    if not merged_data:
        logger.error("No matching data found")
        return
    
    # Limit samples if specified
    if max_samples and max_samples < len(merged_data):
        merged_data = merged_data[:max_samples]
        logger.info(f"Limited to {max_samples} samples")
    
    # Generate fusion descriptions
    results = generate_fusion_descriptions(merged_data, model, tokenizer)
    
    # Save results
    output_data = {
        "experiment_config": {
            "model": MODEL_NAME,
            "strategy": "S2_Fusion",
            "language": language,
            "audio_description_file": audio_description_file,
            "transcription_file": transcription_file,
            "total_samples": len(results),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "fusion_results": results
    }
    
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"S2 fusion experiment completed")
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"Processed {len(results)} samples")
    
    # Print sample results if verbose
    if verbose and results:
        logger.info("\nSample results:")
        for i, result in enumerate(results[:3]):
            logger.info(f"\nSample {i+1} ({result['file_id']}):")
            logger.info(f"Audio Description: {result['audio_description'][:100]}...")
            logger.info(f"Transcription: {result['original_transcription'][:100]}...")
            logger.info(f"Fusion Result: {result['fusion_description'][:100]}...")

def main():
    parser = argparse.ArgumentParser(description="Run S2 strategy fusion experiment")
    parser.add_argument(
        '--audio_description_file', 
        type=str, 
        required=True,
        help="Path to audio description results file (e.g., cloud_baseline_results.json)"
    )
    parser.add_argument(
        '--transcription_file', 
        type=str, 
        required=True,
        help="Path to manifest file with transcriptions (e.g., manifest_audio_text_augmented_v5.json)"
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='experiments/results/s2_fusion_results.json',
        help="Path for output file"
    )
    parser.add_argument(
        '--language', 
        type=str, 
        default='english',
        choices=['english', 'chinese'],
        help="Language for transcriptions"
    )
    parser.add_argument(
        '--max_samples', 
        type=int, 
        default=None,
        help="Maximum number of samples to process"
    )
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help="Print detailed information"
    )
    
    args = parser.parse_args()
    
    run_s2_fusion_experiment(
        audio_description_file=args.audio_description_file,
        transcription_file=args.transcription_file,
        output_file=args.output_file,
        language=args.language,
        max_samples=args.max_samples,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()
