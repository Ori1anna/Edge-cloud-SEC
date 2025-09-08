"""
Evaluation metrics for speech emotion captioning
"""

import numpy as np
from typing import List, Dict, Any
import logging
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk
import re

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class EvaluationMetrics:
    """Evaluation metrics for speech emotion captioning"""
    
    def __init__(self):
        self.smoothing = SmoothingFunction().method1
    
    def compute_bleu(self, references: List[str], hypothesis: str) -> float:
        """Compute BLEU-4 score"""
        try:
            # Clean hypothesis text (remove special tokens)
            hypothesis = hypothesis.replace('<|im_end|>', '').strip()
            
            # Use character-level tokenization for Chinese text
            ref_tokens = [list(ref.lower()) for ref in references]
            hyp_tokens = list(hypothesis.lower())
            
            # Debug logging
            logger.debug(f"Reference tokens: {ref_tokens}")
            logger.debug(f"Hypothesis tokens: {hyp_tokens}")
            
            bleu_score = sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=self.smoothing)
            logger.debug(f"BLEU score: {bleu_score}")
            
            return bleu_score
        except Exception as e:
            logger.error(f"Error computing BLEU: {e}")
            return 0.0
    
    def compute_cider(self, references: List[str], hypothesis: str) -> float:
        """Compute CIDEr score (simplified version)"""
        # This is a simplified CIDEr implementation
        # For full implementation, consider using pycocoevalcap
        try:
            # Use character-level tokenization for Chinese text
            ref_tokens = [list(ref.lower()) for ref in references]
            hyp_tokens = list(hypothesis.lower())
            
            # Simple TF-IDF based scoring
            all_tokens = set()
            for ref in ref_tokens:
                all_tokens.update(ref)
            all_tokens.update(hyp_tokens)
            
            # Compute TF-IDF vectors
            ref_vectors = []
            for ref in ref_tokens:
                vector = [ref.count(token) for token in all_tokens]
                ref_vectors.append(vector)
            
            hyp_vector = [hyp_tokens.count(token) for token in all_tokens]
            
            # Compute cosine similarity
            similarities = []
            for ref_vec in ref_vectors:
                similarity = self._cosine_similarity(ref_vec, hyp_vector)
                similarities.append(similarity)
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception as e:
            logger.error(f"Error computing CIDEr: {e}")
            return 0.0
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def compute_emotion_accuracy(self, 
                                predicted_emotions: List[str], 
                                reference_emotions: List[str]) -> float:
        """Compute emotion classification accuracy"""
        try:
            if len(predicted_emotions) != len(reference_emotions):
                logger.warning("Length mismatch in emotion accuracy computation")
                return 0.0
            
            correct = 0
            total = len(predicted_emotions)
            
            for pred, ref in zip(predicted_emotions, reference_emotions):
                if pred.lower() == ref.lower():
                    correct += 1
            
            return correct / total if total > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error computing emotion accuracy: {e}")
            return 0.0
    
    def compute_detailed_latency_metrics(self, 
                                        latency_data: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Compute simplified latency metrics (mean, min, max only) including CPU, RAM and GPU usage
        
        Args:
            latency_data: List of dictionaries containing detailed latency metrics
            
        Returns:
            Dictionary of simplified latency metrics including CPU, RAM and GPU usage
        """
        try:
            if not latency_data:
                return {}
            
            # Extract individual metrics
            ttft_values = [data.get('ttft', 0) for data in latency_data]
            itps_values = [data.get('itps', 0) for data in latency_data]
            otps_values = [data.get('otps', 0) for data in latency_data]
            oet_values = [data.get('oet', 0) for data in latency_data]
            total_time_values = [data.get('total_time', 0) for data in latency_data]
            input_token_counts = [data.get('input_tokens', 0) for data in latency_data]
            output_token_counts = [data.get('output_tokens', 0) for data in latency_data]
            cpu_percent_values = [data.get('cpu_percent', 0) for data in latency_data]
            ram_gb_values = [data.get('ram_gb', 0) for data in latency_data]
            gpu_util_values = [data.get('gpu_util', 0) for data in latency_data]
            gpu_memory_gb_values = [data.get('gpu_memory_gb', 0) for data in latency_data]
            
            # Convert to numpy arrays for calculations
            ttft_array = np.array(ttft_values)
            itps_array = np.array(itps_values)
            otps_array = np.array(otps_values)
            oet_array = np.array(oet_values)
            total_time_array = np.array(total_time_values)
            input_tokens_array = np.array(input_token_counts)
            output_tokens_array = np.array(output_token_counts)
            cpu_percent_array = np.array(cpu_percent_values)
            ram_gb_array = np.array(ram_gb_values)
            gpu_util_array = np.array(gpu_util_values)
            gpu_memory_gb_array = np.array(gpu_memory_gb_values)
            
            # Calculate simplified metrics (mean, min, max only)
            metrics = {}
            
            # TTFT (Time-to-First-Token) - simplified
            metrics['ttft_mean'] = float(np.mean(ttft_array))
            metrics['ttft_min'] = float(np.min(ttft_array))
            metrics['ttft_max'] = float(np.max(ttft_array))
            
            # ITPS (Input Token Per Second) - simplified
            metrics['itps_mean'] = float(np.mean(itps_array))
            metrics['itps_min'] = float(np.min(itps_array))
            metrics['itps_max'] = float(np.max(itps_array))
            
            # OTPS (Output Token Per Second) - simplified
            metrics['otps_mean'] = float(np.mean(otps_array))
            metrics['otps_min'] = float(np.min(otps_array))
            metrics['otps_max'] = float(np.max(otps_array))
            
            # OET (Output Evaluation Time) - simplified
            metrics['oet_mean'] = float(np.mean(oet_array))
            metrics['oet_min'] = float(np.min(oet_array))
            metrics['oet_max'] = float(np.max(oet_array))
            
            # Total Time - simplified
            metrics['total_time_mean'] = float(np.mean(total_time_array))
            metrics['total_time_min'] = float(np.min(total_time_array))
            metrics['total_time_max'] = float(np.max(total_time_array))
            
            # CPU usage - simplified
            metrics['cpu_percent_mean'] = float(np.mean(cpu_percent_array))
            metrics['cpu_percent_min'] = float(np.min(cpu_percent_array))
            metrics['cpu_percent_max'] = float(np.max(cpu_percent_array))
            
            # RAM usage - simplified
            metrics['ram_gb_mean'] = float(np.mean(ram_gb_array))
            metrics['ram_gb_min'] = float(np.min(ram_gb_array))
            metrics['ram_gb_max'] = float(np.max(ram_gb_array))
            
            # GPU utilization - simplified
            metrics['gpu_util_mean'] = float(np.mean(gpu_util_array))
            metrics['gpu_util_min'] = float(np.min(gpu_util_array))
            metrics['gpu_util_max'] = float(np.max(gpu_util_array))
            
            # GPU memory usage - simplified
            metrics['gpu_memory_gb_mean'] = float(np.mean(gpu_memory_gb_array))
            metrics['gpu_memory_gb_min'] = float(np.min(gpu_memory_gb_array))
            metrics['gpu_memory_gb_max'] = float(np.max(gpu_memory_gb_array))
            
            # Token count statistics - simplified
            metrics['avg_input_tokens'] = float(np.mean(input_tokens_array))
            metrics['avg_output_tokens'] = float(np.mean(output_tokens_array))
            metrics['total_input_tokens'] = int(np.sum(input_tokens_array))
            metrics['total_output_tokens'] = int(np.sum(output_tokens_array))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing detailed latency metrics: {e}")
            return {}
    
    def compute_latency_metrics(self, 
                               latencies: List[float]) -> Dict[str, float]:
        """Compute basic latency statistics (kept for backward compatibility)"""
        try:
            latencies = np.array(latencies)
            
            return {
                'mean_latency': float(np.mean(latencies)),
                'median_latency': float(np.median(latencies)),
                'std_latency': float(np.std(latencies)),
                'min_latency': float(np.min(latencies)),
                'max_latency': float(np.max(latencies)),
                'p95_latency': float(np.percentile(latencies, 95)),
                'p99_latency': float(np.percentile(latencies, 99))
            }
        except Exception as e:
            logger.error(f"Error computing latency metrics: {e}")
            return {}
    
    def compute_all_metrics(self, 
                           predictions: List[str],
                           references: List[List[str]],
                           predicted_emotions: List[str] = None,
                           reference_emotions: List[str] = None,
                           latencies: List[float] = None) -> Dict[str, float]:
        """Compute all evaluation metrics"""
        metrics = {}
        
        # Text quality metrics
        bleu_scores = []
        cider_scores = []
        
        for pred, refs in zip(predictions, references):
            bleu = self.compute_bleu(refs, pred)
            cider = self.compute_cider(refs, pred)
            bleu_scores.append(bleu)
            cider_scores.append(cider)
        
        metrics['bleu'] = np.mean(bleu_scores)
        metrics['cider'] = np.mean(cider_scores)
        
        # Emotion accuracy
        if predicted_emotions and reference_emotions:
            metrics['emotion_accuracy'] = self.compute_emotion_accuracy(
                predicted_emotions, reference_emotions
            )
        
        # Latency metrics
        if latencies:
            latency_metrics = self.compute_latency_metrics(latencies)
            metrics.update(latency_metrics)
        
        return metrics
