#!/usr/bin/env python3
"""
Extract emotion labels from English emotional descriptions using Qwen-2.5-Omni-7B

This script reads generated emotional descriptions from a JSON file,
uses Qwen to extract open-vocabulary emotion labels (1-8 per sample),
and saves them in the format required by MER2024 OV-MER evaluation.

Usage:
    python tools/extract_emotion_labels.py \
        --input_json experiments/results/cloud_mer_en_test1.json \
        --output_csv MERTools/MER2024/ov_store/predict-openset-qwen.csv \
        --model_name Qwen/Qwen2.5-Omni-7B \
        --device cuda:0
"""

import json
import argparse
import logging
import re
import csv
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Emotion extraction prompt (based on OV-MER paper Appendix #5)
# User prompt only - system prompt must remain default for Qwen2.5-Omni
EMOTION_EXTRACTION_USER_PROMPT = """Please assume the role of an expert in the field of emotions. We provide clues that may be related to the emotions of the characters. Based on the provided clues, please identify the emotional states of the main characters. Please separate different emotional categories with commas and output only the clearly identifiable emotional categories in a list format. If none are identified, please output an empty list.

Clues: {description}

Output format: ["emotion1", "emotion2", ...] or []
Output:"""


def load_qwen_model(model_name: str, device: str):
    """
    Load Qwen-2.5-Omni model and processor (text-only mode)
    
    Args:
        model_name: Model name or path
        device: Device to load model on (e.g., 'cuda:0', 'cpu')
    
    Returns:
        model, processor
    """
    logger.info(f"Loading model: {model_name}")
    
    # Load processor
    processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
    
    # Load model (use torch_dtype="auto" as recommended by official docs)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    
    # Disable audio output to save memory (we only need text output)
    model.disable_talker()
    
    model.eval()
    
    logger.info(f"Model loaded successfully on {device}")
    return model, processor


def extract_emotion_labels(
    description: str,
    model,
    processor,
    device: str,
    max_new_tokens: int = 100,
    temperature: float = 0.3
) -> List[str]:
    """
    Extract emotion labels from a description using Qwen model
    
    Args:
        description: English emotional description
        model: Qwen model
        processor: Qwen processor
        device: Device
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (lower = more deterministic)
    
    Returns:
        List of emotion labels (lowercase, deduplicated)
    """
    # Build user prompt
    user_prompt = EMOTION_EXTRACTION_USER_PROMPT.format(description=description)
    
    # Use official Qwen2.5-Omni system prompt (required for proper operation)
    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt}
            ],
        },
    ]
    
    # Apply chat template
    text = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False
    )
    
    # Tokenize (text-only input)
    inputs = processor(
        text=text,
        return_tensors="pt",
        padding=True
    )
    
    # Move to device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
              for k, v in inputs.items()}
    
    # Generate (text-only, no audio output)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            top_p=0.9,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            return_audio=False  # Only return text output
        )
    
    # Decode
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs['input_ids'], output_ids)
    ]
    response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    # Parse JSON response
    labels = parse_emotion_labels(response)
    
    return labels


def parse_emotion_labels(response: str) -> List[str]:
    """
    Parse emotion labels from model response
    
    Handles various formats:
    - JSON array: ["angry", "frustrated"]
    - Comma-separated: angry, frustrated
    - Newline-separated
    
    Args:
        response: Model response text
    
    Returns:
        List of emotion labels (lowercase, deduplicated, max 8)
    """
    labels = []
    
    # Try to extract JSON array
    json_match = re.search(r'\[([^\]]+)\]', response)
    if json_match:
        try:
            # Parse as JSON
            json_str = '[' + json_match.group(1) + ']'
            labels = json.loads(json_str)
            if isinstance(labels, list):
                labels = [str(label).strip().lower() for label in labels]
        except json.JSONDecodeError:
            # Fallback: split by comma
            content = json_match.group(1)
            labels = [item.strip(' "\'').lower() for item in content.split(',')]
    else:
        # Fallback: try comma-separated or newline-separated
        lines = response.strip().split('\n')
        for line in lines:
            if ',' in line:
                labels.extend([item.strip().lower() for item in line.split(',')])
            else:
                label = line.strip().lower()
                if label and not label.startswith(('output', 'result', 'emotion')):
                    labels.append(label)
    
    # Clean up labels
    cleaned_labels = []
    for label in labels:
        # Remove quotes and extra whitespace
        label = label.strip(' "\'')
        # Skip empty or invalid labels
        if label and len(label) > 1 and label.isascii():
            cleaned_labels.append(label)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_labels = []
    for label in cleaned_labels:
        if label not in seen:
            seen.add(label)
            unique_labels.append(label)
    
    # Limit to 8 labels
    unique_labels = unique_labels[:8]
    
    # If empty, return empty list (as per requirements)
    return unique_labels if unique_labels else []


def process_json_file(
    input_json: str,
    output_csv: str,
    model,
    processor,
    device: str
):
    """
    Process all samples in a JSON file and save results to CSV
    
    Args:
        input_json: Path to input JSON file with generated descriptions
        output_csv: Path to output CSV file
        model: Qwen model
        processor: Qwen processor
        device: Device
    """
    # Read input JSON
    logger.info(f"Reading input JSON: {input_json}")
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get samples
    if 'detailed_results' in data:
        samples = data['detailed_results']
    else:
        samples = data if isinstance(data, list) else [data]
    
    logger.info(f"Processing {len(samples)} samples...")
    
    # Process each sample
    results = []
    for idx, sample in enumerate(samples, 1):
        file_id = sample.get('file_id', sample.get('name', f'sample_{idx:08d}'))
        description = sample.get('generated_text', sample.get('description', ''))
        
        if not description:
            logger.warning(f"Sample {idx}/{len(samples)} ({file_id}): No description found, skipping")
            continue
        
        # Extract labels
        try:
            labels = extract_emotion_labels(
                description=description,
                model=model,
                processor=processor,
                device=device
            )
            
            logger.info(f"Sample {idx}/{len(samples)}: {file_id} -> {labels}")
            
            results.append({
                'name': file_id,
                'openset': str(labels)  # Convert to string representation of list
            })
        
        except Exception as e:
            logger.error(f"Sample {idx}/{len(samples)} ({file_id}): Error extracting labels: {e}")
            # Add empty list for failed samples
            results.append({
                'name': file_id,
                'openset': '[]'
            })
    
    # Save to CSV
    logger.info(f"Saving results to: {output_csv}")
    df = pd.DataFrame(results)
    
    # Ensure output directory exists
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save with proper quoting
    df.to_csv(
        output_csv,
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
        encoding='utf-8'
    )
    
    logger.info(f"Successfully saved {len(results)} samples to {output_csv}")
    
    # Print summary statistics
    total_labels = sum([len(eval(r['openset'])) for r in results])
    avg_labels = total_labels / len(results) if results else 0
    empty_count = sum([1 for r in results if r['openset'] == '[]'])
    
    logger.info(f"\nSummary:")
    logger.info(f"  Total samples: {len(results)}")
    logger.info(f"  Total labels: {total_labels}")
    logger.info(f"  Avg labels per sample: {avg_labels:.2f}")
    logger.info(f"  Samples with no labels: {empty_count}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract emotion labels from descriptions using Qwen-2.5-Omni-7B'
    )
    parser.add_argument(
        '--input_json',
        type=str,
        required=True,
        help='Path to input JSON file with generated descriptions'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        required=True,
        help='Path to output CSV file (predict-openset.csv format)'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='Qwen/Qwen2.5-Omni-7B',
        help='Model name or path (default: Qwen/Qwen2.5-Omni-7B)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to use (default: cuda:0)'
    )
    
    args = parser.parse_args()
    
    # Load model
    model, processor = load_qwen_model(args.model_name, args.device)
    
    # Process file
    process_json_file(
        input_json=args.input_json,
        output_csv=args.output_csv,
        model=model,
        processor=processor,
        device=args.device
    )
    
    logger.info("\n✅ Done! You can now run the evaluation script:")
    logger.info(f"\ncd MERTools/MER2024")
    logger.info(f"python main-ov.py calculate_openset_overlap_rate_mer2024 \\")
    logger.info(f"    --gt_csv='ov_store/check-openset.csv' \\")
    logger.info(f"    --pred_csv='ov_store/{Path(args.output_csv).name}' \\")
    logger.info(f"    --synonym_root='ov_store/openset-synonym'")


if __name__ == '__main__':
    main()

