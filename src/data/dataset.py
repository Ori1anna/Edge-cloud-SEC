"""
Dataset loader for unified manifest
"""

import json
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

from .audio_processor import AudioProcessor

logger = logging.getLogger(__name__)


class SpeechEmotionDataset(Dataset):
    """Dataset for speech emotion captioning"""
    
    def __init__(self, 
                 manifest_path: str,
                 audio_processor: AudioProcessor,
                 max_length: int = 512,
                 feature_cache_dir: Optional[str] = None):
        self.manifest_path = manifest_path
        self.audio_processor = audio_processor
        self.max_length = max_length
        self.feature_cache_dir = Path(feature_cache_dir) if feature_cache_dir else None
        
        # Load manifest
        with open(manifest_path, 'r', encoding='utf-8') as f:
            self.manifest = json.load(f)
        
        logger.info(f"Loaded {len(self.manifest)} samples from {manifest_path}")
        
        # Create cache directory
        if self.feature_cache_dir:
            self.feature_cache_dir.mkdir(exist_ok=True)
    
    def __len__(self):
        return len(self.manifest)
    
    def __getitem__(self, idx):
        sample = self.manifest[idx]
        
        # Load audio features
        audio_path = sample['audio_path']
        audio_features = self._load_audio_features(audio_path)
        
        # Prepare text data
        caption = sample.get('caption', '')
        transcription = sample.get('transcription', '')
        
        return {
            'file_id': sample['file_id'],
            'dataset': sample['dataset'],
            'audio_features': audio_features,
            'caption': caption,
            'transcription': transcription,
            'chinese_caption': sample.get('chinese_caption', ''),
            'english_caption': sample.get('english_caption', ''),
            'chinese_transcription': sample.get('chinese_transcription', ''),
            'english_transcription': sample.get('english_transcription', '')
        }
    
    def _load_audio_features(self, audio_path: str) -> torch.Tensor:
        """Load audio features with caching"""
        if self.feature_cache_dir:
            cache_path = self.feature_cache_dir / f"{Path(audio_path).stem}.pt"
            
            if cache_path.exists():
                try:
                    return torch.load(cache_path)
                except Exception as e:
                    logger.warning(f"Failed to load cached features: {e}")
        
        # Extract features
        features = self.audio_processor.extract_features(audio_path)
        
        # Cache features
        if self.feature_cache_dir:
            try:
                torch.save(features, cache_path)
            except Exception as e:
                logger.warning(f"Failed to cache features: {e}")
        
        return features
    
    def get_dataloader(self, 
                      batch_size: int = 8,
                      shuffle: bool = True,
                      num_workers: int = 4):
        """Get PyTorch DataLoader"""
        from torch.utils.data import DataLoader
        
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """Custom collate function for batching"""
        # Pad audio features to same length
        max_feature_length = max(item['audio_features'].shape[-1] for item in batch)
        
        padded_features = []
        for item in batch:
            features = item['audio_features']
            if features.shape[-1] < max_feature_length:
                # Pad with zeros
                padding = torch.zeros(features.shape[0], max_feature_length - features.shape[-1])
                features = torch.cat([features, padding], dim=-1)
            padded_features.append(features)
        
        return {
            'file_ids': [item['file_id'] for item in batch],
            'datasets': [item['dataset'] for item in batch],
            'audio_features': torch.stack(padded_features),
            'captions': [item['caption'] for item in batch],
            'transcriptions': [item['transcription'] for item in batch],
            'chinese_captions': [item['chinese_caption'] for item in batch],
            'english_captions': [item['english_caption'] for item in batch],
            'chinese_transcriptions': [item['chinese_transcription'] for item in batch],
            'english_transcriptions': [item['english_transcription'] for item in batch]
        }
