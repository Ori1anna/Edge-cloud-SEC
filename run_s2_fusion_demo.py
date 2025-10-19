#!/usr/bin/env python3
"""
Demo script to run S2 fusion with your existing results
"""

import os
import sys
import subprocess

def main():
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    s2_script = os.path.join(script_dir, "experiments/runs/run_s2_fusion.py")
    
    # Example command - you can modify these paths
    audio_desc_file = "experiments/results/cloud_mer_en_test9.json"
    transcription_file = "data/processed/mer2024/manifest_audio_text_augmented_v5.json"
    output_file = "experiments/results/s2_fusion_demo.json"
    
    # Check if files exist
    if not os.path.exists(audio_desc_file):
        print(f"Audio description file not found: {audio_desc_file}")
        print("Please provide the correct path to your audio description results file")
        return
    
    if not os.path.exists(transcription_file):
        print(f"Transcription file not found: {transcription_file}")
        print("Please provide the correct path to your manifest file")
        return
    
    # Run S2 fusion
    cmd = [
        sys.executable, s2_script,
        "--audio_description_file", audio_desc_file,
        "--transcription_file", transcription_file,
        "--output_file", output_file,
        "--language", "english",
        "--max_samples", "5",  # Start with just 5 samples for testing
        "--verbose"
    ]
    
    print("Running S2 fusion experiment...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("S2 fusion completed successfully!")
        print(f"Results saved to: {output_file}")
        
        # Print some output for verification
        if result.stdout:
            print("\nOutput:")
            print(result.stdout)
            
    except subprocess.CalledProcessError as e:
        print(f"Error running S2 fusion: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
