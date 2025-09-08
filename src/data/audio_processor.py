"""
Audio processing utilities
"""

import torch
import torchaudio
import librosa
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Audio feature extraction and processing"""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 n_mels: int = 80,
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 feature_type: str = "mel",
                 normalize: bool = True):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.feature_type = feature_type
        self.normalize = normalize
        
    def load_audio(self, file_path: str) -> torch.Tensor:
        """Load audio file and resample if needed"""
        try:
            # Load audio with torchaudio
            waveform, sr = torchaudio.load(file_path)
            
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            return waveform.squeeze()
            
        except Exception as e:
            logger.error(f"Failed to load audio {file_path}: {e}")
            raise
    
    def extract_mel_spectrogram(self, 
                               waveform: torch.Tensor,
                               normalize: bool = True) -> torch.Tensor:
        """Extract mel spectrogram features"""
        # Convert to numpy for librosa
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.numpy()
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Convert to log scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        if normalize:
            mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()
        
        return torch.from_numpy(mel_spec).float()
    
    def extract_mfcc(self, 
                    waveform: torch.Tensor,
                    n_mfcc: int = 13) -> torch.Tensor:
        """Extract MFCC features"""
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.numpy()
        
        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        return torch.from_numpy(mfcc).float()
    
    def extract_features(self, 
                        file_path: str,
                        feature_type: str = None) -> torch.Tensor:
        """
        Extract audio features from file
        
        Args:
            file_path: Path to audio file
            feature_type: Type of features ("mel" or "mfcc"), uses instance default if None
            
        Returns:
            Feature tensor
        """
        waveform = self.load_audio(file_path)
        
        # Use instance feature_type if not provided
        if feature_type is None:
            feature_type = self.feature_type
        
        if feature_type == "mel":
            return self.extract_mel_spectrogram(waveform, normalize=self.normalize)
        elif feature_type == "mfcc":
            return self.extract_mfcc(waveform)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
    
    def pad_or_truncate(self, 
                       features: torch.Tensor,
                       target_length: int) -> torch.Tensor:
        """Pad or truncate features to target length"""
        current_length = features.shape[-1]
        
        if current_length > target_length:
            # Truncate
            return features[:, :target_length]
        elif current_length < target_length:
            # Pad with zeros
            padding = torch.zeros(features.shape[0], target_length - current_length)
            return torch.cat([features, padding], dim=-1)
        else:
            return features
