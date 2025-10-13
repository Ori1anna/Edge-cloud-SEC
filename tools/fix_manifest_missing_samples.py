#!/usr/bin/env python3
"""
Fix missing samples in manifest_audio_only_final.json

This script ensures all 332 samples from final-EMER-reason.csv are included
in the manifest by matching with label-transcription.csv

Data structure:
- final-EMER-reason.csv: name, chinese (caption), english (caption)
- label-transcription.csv: name, chinese (transcription), english (transcription)
- Output manifest: file_id, audio_path, chinese_caption, english_caption, 
                   chinese_transcription, english_transcription
"""

import json
import pandas as pd
from pathlib import Path

def main():
    # Paths
    base_dir = Path('/data/gpfs/projects/punim2341/jiajunlu/edgecloud-sec/data/processed/mer2024')
    reason_csv = base_dir / 'final-EMER-reason.csv'
    label_csv = base_dir / 'label-transcription.csv'
    audio_dir = base_dir / 'audio_16k_mono'
    output_json = base_dir / 'manifest_audio_only_final_fixed.json'
    
    print("=" * 80)
    print("Fixing manifest_audio_only_final.json - Adding missing 20 samples")
    print("=" * 80)
    
    # Read reason CSV (contains captions)
    print(f"\n1. Reading {reason_csv.name}...")
    reason_df = pd.read_csv(reason_csv, encoding='utf-8')
    print(f"   Found {len(reason_df)} samples with captions")
    print(f"   Columns: {reason_df.columns.tolist()}")
    
    # Read label CSV (contains transcriptions)
    print(f"\n2. Reading {label_csv.name}...")
    label_df = pd.read_csv(label_csv, encoding='utf-8')
    print(f"   Found {len(label_df)} samples with transcriptions")
    print(f"   Columns: {label_df.columns.tolist()}")
    
    # Merge data
    print(f"\n3. Merging data...")
    # reason_df: name, chinese (caption), english (caption)
    # label_df: name, chinese (transcription), english (transcription)
    merged_df = reason_df.merge(
        label_df, 
        on='name', 
        how='left',
        suffixes=('_caption', '_transcription')
    )
    print(f"   Merged samples: {len(merged_df)}")
    
    # Check for missing matches
    missing_transcriptions = merged_df[merged_df['chinese_transcription'].isna()]
    if len(missing_transcriptions) > 0:
        print(f"   ⚠️  WARNING: {len(missing_transcriptions)} samples have no transcription:")
        for idx, row in missing_transcriptions.head(5).iterrows():
            print(f"      - {row['name']}")
    
    # Build manifest
    print(f"\n4. Building manifest...")
    manifest = []
    skipped_no_audio = []
    samples_no_transcription = []
    
    for idx, row in merged_df.iterrows():
        sample_name = row['name']
        
        # Build audio path
        audio_file = audio_dir / f"{sample_name}.wav"
        
        # Check if audio file exists (ONLY skip if no audio file)
        if not audio_file.exists():
            skipped_no_audio.append(sample_name)
            continue
        
        # Handle missing transcription - use empty string instead of skipping
        chinese_transcription = row.get('chinese_transcription', '')
        english_transcription = row.get('english_transcription', '')
        
        if pd.isna(chinese_transcription):
            chinese_transcription = ''
            samples_no_transcription.append(sample_name)
        if pd.isna(english_transcription):
            english_transcription = ''
        
        # Create manifest entry (matching original manifest.json format)
        entry = {
            "file_id": sample_name,
            "dataset": "mer2024",
            "audio_path": str(audio_file),
            "chinese_caption": row['chinese_caption'],      # Caption from final-EMER-reason.csv
            "english_caption": row['english_caption'],      # Caption from final-EMER-reason.csv
            "chinese_transcription": chinese_transcription,  # Transcription from label-transcription.csv (or empty)
            "english_transcription": english_transcription,  # Transcription from label-transcription.csv (or empty)
        }
        
        manifest.append(entry)
    
    print(f"   ✅ Built manifest with {len(manifest)} samples")
    
    if samples_no_transcription:
        print(f"   ℹ️  {len(samples_no_transcription)} samples have empty transcription (kept in manifest)")
    if skipped_no_audio:
        print(f"   ⚠️  Skipped {len(skipped_no_audio)} samples (no audio file)")
    
    # Save manifest
    print(f"\n5. Saving to {output_json.name}...")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    print(f"   ✅ Saved successfully")
    
    # Summary
    print(f"\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Input:")
    print(f"  - final-EMER-reason.csv: {len(reason_df)} samples (with captions)")
    print(f"  - label-transcription.csv: {len(label_df)} samples (with transcriptions)")
    print(f"\nOutput:")
    print(f"  - manifest_audio_only_final_fixed.json: {len(manifest)} samples")
    print(f"\nDetails:")
    print(f"  - Samples with transcription: {len(manifest) - len(samples_no_transcription)}")
    print(f"  - Samples with empty transcription: {len(samples_no_transcription)}")
    print(f"  - Skipped (no audio file): {len(skipped_no_audio)}")
    
    # Check if we got all 332 samples
    expected = len(reason_df)
    actual = len(manifest)
    diff = expected - actual
    
    print(f"\nComparison:")
    if diff == 0:
        print(f"  ✅ Perfect! All {expected} samples included!")
    else:
        print(f"  ⚠️  Missing {diff} samples")
        
        # Find missing samples
        manifest_ids = set([item['file_id'] for item in manifest])
        reason_ids = set(reason_df['name'].values)
        missing = reason_ids - manifest_ids
        
        print(f"\n  Missing sample IDs ({len(missing)}):")
        for sample_id in sorted(missing):
            # Check why it's missing
            if sample_id in skipped_no_audio:
                reason = "no audio file"
            else:
                reason = "unknown"
            print(f"    - {sample_id} ({reason})")

if __name__ == '__main__':
    main()

