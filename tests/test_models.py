"""
Tests for model implementations
"""

import pytest
import torch
import tempfile
import os
from unittest.mock import Mock, patch

from src.models.edge_model import EdgeModel
from src.models.cloud_model import CloudModel
from src.models.speculative_decoder import SpeculativeDecoder


class TestEdgeModel:
    """Test edge model functionality"""
    
    def test_initialization(self):
        """Test edge model initialization"""
        # Mock the model loading to avoid downloading
        with patch('src.models.edge_model.AutoModelForCausalLM.from_pretrained') as mock_model, \
             patch('src.models.edge_model.AutoTokenizer.from_pretrained') as mock_tokenizer:
            
            mock_model.return_value = Mock()
            mock_tokenizer.return_value = Mock()
            
            model = EdgeModel()
            assert model.model is not None
            assert model.tokenizer is not None
    
    def test_compute_uncertainty(self):
        """Test uncertainty computation"""
        model = EdgeModel()
        
        # Mock log probabilities
        log_probs = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])
        tokens = [1, 2]
        
        uncertainty = model.compute_uncertainty(tokens, log_probs)
        
        assert 'token_log_probs' in uncertainty
        assert 'entropy' in uncertainty
        assert 'margin' in uncertainty
        assert len(uncertainty['entropy']) == 2
        assert len(uncertainty['margin']) == 2


class TestCloudModel:
    """Test cloud model functionality"""
    
    def test_initialization(self):
        """Test cloud model initialization"""
        with patch('src.models.cloud_model.AutoModelForCausalLM.from_pretrained') as mock_model, \
             patch('src.models.cloud_model.AutoTokenizer.from_pretrained') as mock_tokenizer:
            
            mock_model.return_value = Mock()
            mock_tokenizer.return_value = Mock()
            
            model = CloudModel()
            assert model.model is not None
            assert model.tokenizer is not None
            assert hasattr(model, 'kv_cache')
            assert hasattr(model, 'prefix_cache')


class TestSpeculativeDecoder:
    """Test speculative decoder functionality"""
    
    def test_initialization(self):
        """Test decoder initialization"""
        with patch('src.models.edge_model.EdgeModel') as mock_edge, \
             patch('src.models.cloud_model.CloudModel') as mock_cloud:
            
            mock_edge.return_value = Mock()
            mock_cloud.return_value = Mock()
            
            config = {
                'entropy_threshold': 2.0,
                'margin_threshold': 0.1,
                'log_prob_threshold': -2.0,
                'content_rules': {
                    'verify_numbers': True,
                    'verify_dates': True,
                    'verify_names': True
                }
            }
            
            decoder = SpeculativeDecoder(mock_edge(), mock_cloud(), config)
            
            assert decoder.entropy_threshold == 2.0
            assert decoder.margin_threshold == 0.1
            assert decoder.log_prob_threshold == -2.0
            assert decoder.content_rules['verify_numbers'] is True
    
    def test_content_rules(self):
        """Test content-aware verification rules"""
        with patch('src.models.edge_model.EdgeModel') as mock_edge, \
             patch('src.models.cloud_model.CloudModel') as mock_cloud:
            
            mock_edge.return_value = Mock()
            mock_cloud.return_value = Mock()
            
            config = {
                'content_rules': {
                    'verify_numbers': True,
                    'verify_dates': True,
                    'verify_names': True
                }
            }
            
            decoder = SpeculativeDecoder(mock_edge(), mock_cloud(), config)
            
            # Mock tokenizer
            decoder.edge_model.tokenizer.decode = lambda x: " ".join([str(t) for t in x])
            
            # Test number detection
            assert decoder._check_content_rules([1, 2, 3]) is False  # No numbers in text
            
            # Test name detection
            assert decoder._check_content_rules([1, 2, 3]) is False  # No names in text
