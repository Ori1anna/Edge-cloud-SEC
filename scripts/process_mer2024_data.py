#!/usr/bin/env python3
"""
Process MER2024 data to unified format
"""

import json
import os
import pandas as pd
from pathlib import Path

def process_mer2024_data():
    """Process MER2024 data to unified format"""
    
    # Load emotion descriptions
    emotion_file = "data/processed/mer2024/final-EMER-reason.csv"
    if not os.path.exists(emotion_file):
        print(f"Error: Emotion file not found: {emotion_file}")
        return
    
    emotion_df = pd.read_csv(emotion_file)
    print(f"Loaded {len(emotion_df)} emotion descriptions")
    
    # Load transcriptions
    transcription_file = "data/processed/mer2024/label-transcription.csv"
    if not os.path.exists(transcription_file):
        print(f"Error: Transcription file not found: {transcription_file}")
        return
    
    transcription_df = pd.read_csv(transcription_file)
    print(f"Loaded {len(transcription_df)} transcriptions")
    
    # Create unified manifest
    manifest = []
    index = 0
    
    for _, row in emotion_df.iterrows():
        sample_id = row['name']
        
        # Find corresponding transcription
        trans_row = transcription_df[transcription_df['name'] == sample_id]
        if len(trans_row) == 0:
            print(f"Warning: No transcription found for {sample_id}")
            continue
        
        # Get Chinese and English captions (emotion descriptions)
        chinese_caption = row['chinese']
        english_caption = row['english']
        if pd.isna(chinese_caption) or chinese_caption.strip() == '':
            print(f"Warning: No Chinese caption for {sample_id}")
            continue
        
        # Get Chinese and English transcriptions
        chinese_trans = trans_row.iloc[0]['chinese']
        english_trans = trans_row.iloc[0]['english']
        if pd.isna(chinese_trans) or chinese_trans.strip() == '':
            print(f"Warning: No Chinese transcription for {sample_id}")
            continue
        
        # Check if audio file exists
        audio_path = f"data/processed/mer2024/audio_16k_mono/{sample_id}.wav"
        if not os.path.exists(audio_path):
            print(f"Warning: Audio file not found: {audio_path}")
            continue
        
        manifest.append({
            "file_id": sample_id,
            "dataset": "mer2024",
            "audio_path": audio_path,
            "caption": chinese_caption.strip(),
            "transcription": chinese_trans.strip(),
            "chinese_caption": chinese_caption.strip(),
            "english_caption": english_caption.strip() if not pd.isna(english_caption) else "",
            "chinese_transcription": chinese_trans.strip(),
            "english_transcription": english_trans.strip() if not pd.isna(english_trans) else "",
            "index": index
        })
        index += 1
    
    # Save manifest
    output_file = "data/processed/mer2024/manifest.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    print(f"Processed {len(manifest)} MER2024 samples")
    print(f"Manifest saved to: {output_file}")

if __name__ == "__main__":
    process_mer2024_data()
