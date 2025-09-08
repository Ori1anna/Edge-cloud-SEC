#!/usr/bin/env python3
"""
Data validation script
Validate the completeness and usability of unified(mer_secap) dataset
"""

import json
import pandas as pd
import numpy as np
import os
import time
from pathlib import Path
from collections import Counter
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_manifest_structure(manifest_path: str):
    """Validate manifest file structure"""
    logger.info("=== Validating manifest file structure ===")
    
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"✅ Successfully loaded manifest file")
        logger.info(f"   Total samples: {len(data)}")
        
        # Check required fields
        required_fields = ['file_id', 'dataset', 'audio_path', 'caption', 'transcription']
        sample = data[0]
        missing_fields = [field for field in required_fields if field not in sample]
        
        if missing_fields:
            logger.error(f"❌ Missing required fields: {missing_fields}")
            return False
        else:
            logger.info(f"✅ All required fields exist")
            logger.info(f"   Field list: {list(sample.keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load manifest file: {e}")
        return False


def analyze_dataset_distribution(data):
    """Analyze dataset distribution"""
    logger.info("=== Analyzing dataset distribution ===")
    
    df = pd.DataFrame(data)
    
    # Dataset distribution
    dataset_counts = df['dataset'].value_counts()
    logger.info("Dataset distribution:")
    for dataset, count in dataset_counts.items():
        percentage = count / len(df) * 100
        logger.info(f"   {dataset}: {count} samples ({percentage:.1f}%)")
    
    # Check missing values
    missing_data = df.isnull().sum()
    missing_fields = missing_data[missing_data > 0]
    
    if len(missing_fields) > 0:
        logger.warning(f"⚠️  Found missing values:")
        for field, count in missing_fields.items():
            logger.warning(f"   {field}: {count} missing values")
    else:
        logger.info("✅ No missing values")
    
    return df


def validate_audio_files(data):
    """Validate audio file existence"""
    logger.info("=== Validating audio files ===")
    
    existing_files = 0
    missing_files = 0
    missing_paths = []
    
    for i, sample in enumerate(data):
        audio_path = sample['audio_path']
        if os.path.exists(audio_path):
            existing_files += 1
        else:
            missing_files += 1
            missing_paths.append(audio_path)
            if missing_files <= 5:  # Only log first 5 missing files
                logger.warning(f"   Missing file: {audio_path}")
    
    logger.info(f"Audio file statistics:")
    logger.info(f"   Existing: {existing_files}")
    logger.info(f"   Missing: {missing_files}")
    logger.info(f"   Existence rate: {existing_files/(existing_files+missing_files)*100:.1f}%")
    
    if missing_files == 0:
        logger.info("✅ All audio files exist")
        return True
    else:
        logger.error(f"❌ {missing_files} audio files are missing")
        return False


def analyze_text_content(data):
    """Analyze text content"""
    logger.info("=== Analyzing text content ===")
    
    df = pd.DataFrame(data)
    
    # Calculate text lengths
    df['caption_length'] = df['caption'].str.len()
    df['transcription_length'] = df['transcription'].str.len()
    
    # Overall statistics
    logger.info("Caption length statistics:")
    logger.info(f"   Average length: {df['caption_length'].mean():.1f} characters")
    logger.info(f"   Median length: {df['caption_length'].median():.1f} characters")
    logger.info(f"   Min length: {df['caption_length'].min()} characters")
    logger.info(f"   Max length: {df['caption_length'].max()} characters")
    
    logger.info("Transcription length statistics:")
    logger.info(f"   Average length: {df['transcription_length'].mean():.1f} characters")
    logger.info(f"   Median length: {df['transcription_length'].median():.1f} characters")
    logger.info(f"   Min length: {df['transcription_length'].min()} characters")
    logger.info(f"   Max length: {df['transcription_length'].max()} characters")
    
    # Analysis by dataset
    logger.info("Analysis by dataset:")
    for dataset in ['secap', 'mer2024']:
        subset = df[df['dataset'] == dataset]
        logger.info(f"   {dataset.upper()}:")
        logger.info(f"     Average caption length: {subset['caption_length'].mean():.1f} characters")
        logger.info(f"     Average transcription length: {subset['transcription_length'].mean():.1f} characters")
    
    return df


def test_audio_processing(data, num_samples=3):
    """Test audio processing functionality"""
    logger.info("=== Testing audio processing ===")
    
    try:
        from src.data.audio_processor import AudioProcessor
        processor = AudioProcessor()
        logger.info("✅ Successfully imported AudioProcessor")
        
        # Test several samples
        for i in range(min(num_samples, len(data))):
            sample = data[i]
            audio_path = sample['audio_path']
            
            try:
                start_time = time.time()
                features = processor.extract_features(audio_path, 'mel')
                processing_time = time.time() - start_time
                
                logger.info(f"   Sample {i+1}: {sample['file_id']}")
                logger.info(f"     Feature shape: {features.shape}")
                logger.info(f"     Processing time: {processing_time:.3f} seconds")
                
            except Exception as e:
                logger.error(f"   ❌ Failed to process sample {sample['file_id']}: {e}")
                return False
        
        logger.info("✅ Audio processing test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to import AudioProcessor: {e}")
        return False


def test_dataset_loader(data, num_samples=2):
    """Test dataset loader"""
    logger.info("=== Testing dataset loader ===")
    
    try:
        from src.data.dataset import SpeechEmotionDataset
        from src.data.audio_processor import AudioProcessor
        
        # Create temporary manifest for testing
        test_data = data[:num_samples]
        test_manifest_path = "experiments/test_manifest.json"
        
        with open(test_manifest_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        # Initialize components
        processor = AudioProcessor()
        dataset = SpeechEmotionDataset(
            manifest_path=test_manifest_path,
            audio_processor=processor,
            feature_cache_dir='experiments/cache'
        )
        
        logger.info(f"✅ Successfully created dataset loader")
        logger.info(f"   Test samples: {len(dataset)}")
        
        # Test single sample loading
        sample = dataset[0]
        logger.info(f"   Single sample fields: {list(sample.keys())}")
        logger.info(f"   Audio feature shape: {sample['audio_features'].shape}")
        
        # Test batch processing
        dataloader = dataset.get_dataloader(batch_size=2, shuffle=False)
        batch = next(iter(dataloader))
        logger.info(f"   Batch shape: {batch['audio_features'].shape}")
        logger.info(f"   Batch File IDs: {batch['file_ids']}")
        
        # Clean up temporary file
        os.remove(test_manifest_path)
        
        logger.info("✅ Dataset loader test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Dataset loader test failed: {e}")
        return False


def main():
    """Main function"""
    logger.info("Starting data validation...")
    
    manifest_path = "data/processed/unified(mer_secap)/unified_manifest.json"
    
    # 1. Validate manifest structure
    if not validate_manifest_structure(manifest_path):
        logger.error("❌ Manifest structure validation failed")
        return
    
    # 2. Load data
    with open(manifest_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 3. Analyze dataset distribution
    df = analyze_dataset_distribution(data)
    
    # 4. Validate audio files
    audio_valid = validate_audio_files(data)
    
    # 5. Analyze text content
    df = analyze_text_content(data)
    
    # 6. Test audio processing
    audio_processing_valid = test_audio_processing(data)
    
    # 7. Test dataset loader
    dataset_loader_valid = test_dataset_loader(data)
    
    # Summary
    logger.info("=== Validation Summary ===")
    logger.info(f"✅ Manifest structure: Passed")
    logger.info(f"✅ Dataset distribution: Passed")
    logger.info(f"{'✅' if audio_valid else '❌'} Audio files: {'Passed' if audio_valid else 'Failed'}")
    logger.info(f"✅ Text content: Passed")
    logger.info(f"{'✅' if audio_processing_valid else '❌'} Audio processing: {'Passed' if audio_processing_valid else 'Failed'}")
    logger.info(f"{'✅' if dataset_loader_valid else '❌'} Dataset loader: {'Passed' if dataset_loader_valid else 'Failed'}")
    
    if all([audio_valid, audio_processing_valid, dataset_loader_valid]):
        logger.info("🎉 All validations passed! Dataset is ready.")
    else:
        logger.error("❌ Some validations failed, please check the issues.")


if __name__ == "__main__":
    main()
