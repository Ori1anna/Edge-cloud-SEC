#!/usr/bin/env python3
"""
Process SECap data to unified format
"""

import json
import os
from pathlib import Path

def process_secap_data():
    """Process SECap data to unified format"""
    
    # Load captions
    captions_file = "data/processed/secap/fid2captions.json"
    with open(captions_file, 'r', encoding='utf-8') as f:
        captions = json.load(f)
    
    # Load transcriptions (if available)
    transcriptions_file = "data/processed/secap/text.txt"
    transcriptions = {}
    if os.path.exists(transcriptions_file):
        with open(transcriptions_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Split by first space to separate file_id and transcription
                    first_space_idx = line.find(' ')
                    if first_space_idx != -1:
                        file_id = line[:first_space_idx]
                        transcription = line[first_space_idx + 1:]
                        transcriptions[file_id] = transcription
    
    # Create unified manifest
    manifest = []
    index = 0
    
    for file_id, caption in captions.items():
        # Check if audio file exists
        audio_path = f"data/processed/secap/wav_16k/{file_id}.wav"
        if not os.path.exists(audio_path):
            print(f"Warning: Audio file not found: {audio_path}")
            continue
        
        # Get transcription (use caption as fallback if not available)
        transcription = transcriptions.get(file_id, caption)
        
        manifest.append({
            "file_id": file_id,
            "dataset": "secap",
            "audio_path": audio_path,
            "caption": caption,
            "transcription": transcription,
            "chinese_caption": caption,
            "english_caption": "",
            "chinese_transcription": transcription,
            "english_transcription": "",
            "index": index
        })
        index += 1
    
    # Save manifest
    output_file = "data/processed/secap/manifest.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    print(f"Processed {len(manifest)} SECap samples")
    print(f"Manifest saved to: {output_file}")

if __name__ == "__main__":
    process_secap_data()
