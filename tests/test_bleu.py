#!/usr/bin/env python3
"""
Test script to debug BLEU calculation issues
"""

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

# Test cases from our results
test_cases = [
    {
        "prediction": "听起来像是有些悲伤呢。你怎么突然问这个呀？",
        "reference": "悲伤逆流成河"
    },
    {
        "prediction": "悲伤的。",
        "reference": "伤心难过，声音颤抖，情绪激动失望"
    },
    {
        "prediction": "听起来这个说话者很悲伤。",
        "reference": "伤心难过，又无能为力"
    }
]

def test_nltk_tokenization():
    """Test how NLTK tokenizes Chinese text"""
    print("=== Testing NLTK Tokenization ===")
    
    for i, case in enumerate(test_cases):
        print(f"\nCase {i+1}:")
        print(f"Prediction: {case['prediction']}")
        print(f"Reference: {case['reference']}")
        
        # NLTK tokenization
        pred_tokens = word_tokenize(case['prediction'].lower())
        ref_tokens = word_tokenize(case['reference'].lower())
        
        print(f"NLTK Pred tokens: {pred_tokens}")
        print(f"NLTK Ref tokens: {ref_tokens}")
        
        # Character-level tokenization (better for Chinese)
        pred_chars = list(case['prediction'].lower())
        ref_chars = list(case['reference'].lower())
        
        print(f"Char Pred tokens: {pred_chars}")
        print(f"Char Ref tokens: {ref_chars}")
        
        # Calculate BLEU with both methods
        smoothing = SmoothingFunction().method1
        
        # NLTK tokenization BLEU
        nltk_bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
        
        # Character-level BLEU
        char_bleu = sentence_bleu([ref_chars], pred_chars, smoothing_function=smoothing)
        
        print(f"NLTK BLEU: {nltk_bleu:.4f}")
        print(f"Char BLEU: {char_bleu:.4f}")

def test_manual_bleu():
    """Test manual BLEU calculation"""
    print("\n=== Manual BLEU Analysis ===")
    
    case = test_cases[0]
    pred = case['prediction']
    ref = case['reference']
    
    # Character-level tokens
    pred_chars = list(pred.lower())
    ref_chars = list(ref.lower())
    
    print(f"Prediction chars: {pred_chars}")
    print(f"Reference chars: {ref_chars}")
    
    # Find common characters
    common_chars = set(pred_chars) & set(ref_chars)
    print(f"Common characters: {common_chars}")
    
    # Count matches
    matches = 0
    for char in ref_chars:
        if char in pred_chars:
            matches += 1
    
    precision = matches / len(pred_chars) if pred_chars else 0
    recall = matches / len(ref_chars) if ref_chars else 0
    
    print(f"Matches: {matches}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

if __name__ == "__main__":
    test_nltk_tokenization()
    test_manual_bleu()
