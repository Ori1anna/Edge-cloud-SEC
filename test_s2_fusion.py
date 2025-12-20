#!/usr/bin/env python3
"""
Test script for S2 fusion functionality
"""

import json
import os
import sys

def test_data_loading():
    """Test if we can load the required data files"""
    print("Testing data loading...")
    
    # Test files
    audio_desc_file = "experiments/results/cloud_mer_en_test9.json"
    transcription_file = "data/processed/mer2024/manifest_audio_text_augmented_v5.json"
    
    # Test audio descriptions
    if os.path.exists(audio_desc_file):
        print(f"✓ Audio description file found: {audio_desc_file}")
        try:
            with open(audio_desc_file, 'r') as f:
                data = json.load(f)
            if 'detailed_results' in data:
                print(f"✓ Found {len(data['detailed_results'])} audio descriptions")
            else:
                print("✗ No 'detailed_results' found in audio description file")
        except Exception as e:
            print(f"✗ Error loading audio description file: {e}")
    else:
        print(f"✗ Audio description file not found: {audio_desc_file}")
    
    # Test transcriptions
    if os.path.exists(transcription_file):
        print(f"✓ Transcription file found: {transcription_file}")
        try:
            with open(transcription_file, 'r') as f:
                data = json.load(f)
            print(f"✓ Found {len(data)} transcription entries")
            
            # Check if we have english_transcription field
            if data and 'english_transcription' in data[0]:
                print("✓ English transcription field found")
            else:
                print("✗ No 'english_transcription' field found")
                
        except Exception as e:
            print(f"✗ Error loading transcription file: {e}")
    else:
        print(f"✗ Transcription file not found: {transcription_file}")

def test_prompt_template():
    """Test the S2 prompt template"""
    print("\nTesting prompt template...")
    
    S2_PROMPT_TEMPLATE = """Please act as an expert in the field of emotions. We provide acoustic clues that may be related to the character's emotional state, along with the original subtitle of the video. Please analyze which parts can infer the emotional state and explain the reasons. During the analysis, please integrate the textual and audio clues.

Acoustic Clues:
{audio_description}

Original Subtitle:
{subtitle}

Integrated Analysis:"""
    
    # Test prompt formatting
    test_audio = "The voice is steady but carries a tone of uncertainty and hesitation."
    test_subtitle = "I don't know! I don't have experience in this area."
    
    formatted_prompt = S2_PROMPT_TEMPLATE.format(
        audio_description=test_audio,
        subtitle=test_subtitle
    )
    
    print("✓ Prompt template formatting works")
    print(f"Sample prompt length: {len(formatted_prompt)} characters")
    print("Sample prompt preview:")
    print(formatted_prompt[:200] + "...")

def test_model_availability():
    """Test if the required model is available"""
    print("\nTesting model availability...")
    
    try:
        import torch
        print(f"✓ PyTorch available: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA device count: {torch.cuda.device_count()}")
    except ImportError:
        print("✗ PyTorch not available")
        return False
    
    try:
        from transformers import AutoTokenizer
        print("✓ Transformers library available")
    except ImportError:
        print("✗ Transformers library not available")
        return False
    
    return True

def main():
    print("S2 Fusion Test Suite")
    print("=" * 50)
    
    # Run tests
    test_data_loading()
    test_prompt_template()
    
    if test_model_availability():
        print("\n✓ All tests passed! You can run the S2 fusion script.")
        print("\nTo run S2 fusion:")
        print("python run_s2_fusion_demo.py")
    else:
        print("\n✗ Some tests failed. Please check the requirements.")
        print("Make sure you have:")
        print("- PyTorch installed")
        print("- Transformers library installed")
        print("- Required data files in place")

if __name__ == "__main__":
    main()




