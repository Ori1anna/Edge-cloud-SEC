#!/usr/bin/env python3
"""
Simple test script to debug the baseline experiment
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_step_by_step():
    """Test each component step by step"""
    
    print("Step 1: Testing config loading...")
    try:
        from src.utils.config import load_config
        config = load_config("configs/default.yaml")
        print("✓ Config loaded successfully")
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return
    
    print("Step 2: Testing manifest loading...")
    try:
        manifest_path = config['data']['train_path']
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        print(f"✓ Manifest loaded successfully, {len(manifest)} samples")
    except Exception as e:
        print(f"✗ Manifest loading failed: {e}")
        return
    
    print("Step 3: Testing audio processor...")
    try:
        from src.data.audio_processor import AudioProcessor
        audio_processor = AudioProcessor(**config['audio'])
        print("✓ Audio processor initialized successfully")
    except Exception as e:
        print(f"✗ Audio processor failed: {e}")
        return
    
    print("Step 4: Testing edge model...")
    try:
        from src.models.edge_model import EdgeModel
        print("✓ Edge model imported successfully")
        # Don't initialize yet as it loads the model
        print("  (Model loading will be tested in full experiment)")
    except Exception as e:
        print(f"✗ Edge model import failed: {e}")
        return
    
    print("Step 5: Testing evaluation metrics...")
    try:
        from src.evaluation.metrics import EvaluationMetrics
        metrics = EvaluationMetrics()
        print("✓ Evaluation metrics initialized successfully")
    except Exception as e:
        print(f"✗ Evaluation metrics failed: {e}")
        return
    
    print("\nAll components imported successfully!")
    print("Ready to run full experiment.")

if __name__ == "__main__":
    test_step_by_step()
