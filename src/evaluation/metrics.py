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

# BERTScore imports
try:
    from bert_score import score as bert_score
    from bert_score import BERTScorer
    BERTSCORE_AVAILABLE = True
    logger.info("BERTScore is available")
except ImportError:
    BERTSCORE_AVAILABLE = False
    logger.warning("BERTScore not available. Install with: pip install bert-score")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class EvaluationMetrics:
    """Evaluation metrics for speech emotion captioning"""
    
    def __init__(self):
        self.smoothing = SmoothingFunction().method1
        self.bertscorer_cache = {}  # Cache for BERTScorer objects
    
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
    
    def compute_bertscore(self, references: List[str], hypothesis: str, language: str = "chinese") -> Dict[str, float]:
        """
        Compute BERTScore for text
        
        Args:
            references: List of reference texts
            hypothesis: Generated text
            language: Language parameter from command line ("chinese", "english", etc.)
            
        Returns:
            Dictionary containing Precision, Recall, and F1 scores
        """
        if not BERTSCORE_AVAILABLE:
            logger.warning("BERTScore not available, returning zero scores")
            return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}
        
        try:
            # Convert language parameter to BERTScore standard language codes
            # Official BERTScore expects: "zh" for Chinese, "en" for English
            if language.lower() in ["chinese", "zh"]:
                bertscore_lang = "zh"
                # model_type = "bert-base-chinese"
                model_type = "hfl/chinese-roberta-wwm-ext-large"
            elif language.lower() in ["english", "en"]:
                bertscore_lang = "en"
                model_type = "roberta-large"
            else:
                # Default to Chinese for other languages
                logger.warning(f"Unknown language '{language}', defaulting to Chinese")
                bertscore_lang = "zh"
                model_type = "bert-base-chinese"
            
            # Clean hypothesis text (remove special tokens)
            hypothesis = hypothesis.replace('<|im_end|>', '').strip()
            
            # Clean reference texts
            clean_references = [ref.replace('<|im_end|>', '').strip() for ref in references]
            
            # Skip empty texts
            if not hypothesis.strip() or not any(ref.strip() for ref in clean_references):
                logger.warning("Empty hypothesis or references, returning zero BERTScore")
                return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}
            
            # Use the best reference (closest length or first non-empty)
            best_reference = None
            for ref in clean_references:
                if ref.strip():
                    best_reference = ref.strip()
                    break
            
            if not best_reference:
                logger.warning("No valid reference found, returning zero BERTScore")
                return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}
            
            logger.debug(f"Computing BERTScore for hypothesis: '{hypothesis}'")
            logger.debug(f"Computing BERTScore for reference: '{best_reference}'")
            logger.debug(f"Using model: {model_type}, language: {bertscore_lang}")
            
            # Check if we have enough references for IDF calculation
            # IDF requires multiple reference sentences to be meaningful
            # For single reference, disable IDF to avoid NaN/invalid scores
            use_idf = len(clean_references) >= 3  # Need at least 3 references for reliable IDF
            
            if use_idf:
                logger.debug("Enabling IDF weighting (sufficient references available)")
            else:
                logger.debug("Disabling IDF weighting (insufficient references for reliable IDF)")
            
            P, R, F1 = bert_score(
                [hypothesis],  # candidates
                [best_reference],  # references
                model_type=model_type,
                lang=bertscore_lang,
                rescale_with_baseline=True,  # Use rescaling for better human correlation
                idf=use_idf,  # Conditionally enable IDF based on reference count
                verbose=False
            )
            
            # Convert to float and return
            result = {
                "bertscore_precision": float(P.item()),
                "bertscore_recall": float(R.item()),
                "bertscore_f1": float(F1.item())
            }
            
            logger.debug(f"BERTScore results: P={result['bertscore_precision']:.4f}, "
                        f"R={result['bertscore_recall']:.4f}, F1={result['bertscore_f1']:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error computing BERTScore: {e}")
            return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}
    
    def compute_batch_bertscore(self, candidates: List[str], references_list: List[List[str]], language: str = "chinese") -> List[Dict[str, float]]:
        """
        Compute BERTScore for a batch of candidates and references using corpus-level IDF
        
        Args:
            candidates: List of generated texts
            references_list: List of reference text lists (one list per candidate)
            language: Language parameter from command line ("chinese", "english", etc.)
            
        Returns:
            List of dictionaries containing Precision, Recall, and F1 scores for each candidate
        """
        if not BERTSCORE_AVAILABLE:
            logger.warning("BERTScore not available, returning zero scores")
            return [{"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0} for _ in candidates]
        
        try:
            # Convert language parameter to BERTScore standard language codes
            if language.lower() in ["chinese", "zh"]:
                bertscore_lang = "zh"
                model_type = "bert-base-chinese"
            elif language.lower() in ["english", "en"]:
                bertscore_lang = "en"
                model_type = "roberta-large"
            else:
                logger.warning(f"Unknown language '{language}', defaulting to Chinese")
                bertscore_lang = "zh"
                model_type = "bert-base-chinese"
            
            # Clean texts
            clean_candidates = [cand.replace('<|im_end|>', '').strip() for cand in candidates]
            clean_references_list = []
            for refs in references_list:
                clean_refs = [ref.replace('<|im_end|>', '').strip() for ref in refs if ref.strip()]
                clean_references_list.append(clean_refs)
            
            # Flatten all references for corpus-level IDF calculation
            all_references = []
            for refs in clean_references_list:
                all_references.extend(refs)
            
            logger.info(f"Computing batch BERTScore with corpus-level IDF for {len(candidates)} candidates")
            logger.info(f"Using model: {model_type}, language: {bertscore_lang}")
            logger.info(f"Total references for IDF: {len(all_references)}")
            logger.info(f"Sample candidates: {clean_candidates[:2] if clean_candidates else 'None'}")
            logger.info(f"Sample references: {clean_references_list[:2] if clean_references_list else 'None'}")
            
            # Create BERTScorer with corpus-level IDF
            cache_key = f"{model_type}_{bertscore_lang}"
            if cache_key not in self.bertscorer_cache:
                try:
                    logger.info(f"Initializing BERTScorer with model_type={model_type}, lang={bertscore_lang}")
                    self.bertscorer_cache[cache_key] = BERTScorer(
                        model_type=model_type,
                        lang=bertscore_lang,
                        rescale_with_baseline=True,  # CRITICAL: Enable rescaling for better human correlation
                        idf=True,  # Enable IDF with corpus-level calculation
                        idf_sents=all_references,  # Use all references for IDF estimation
                        verbose=False  # Reduce logging noise
                    )
                    logger.info(f"Successfully created BERTScorer cache for {cache_key}")
                except Exception as e:
                    logger.error(f"Failed to create BERTScorer: {e}")
                    logger.error(f"Model type: {model_type}, Language: {bertscore_lang}")
                    # Fallback to simple scoring without IDF
                    try:
                        logger.info("Trying fallback BERTScorer without IDF...")
                        self.bertscorer_cache[cache_key] = BERTScorer(
                            model_type=model_type,
                            lang=bertscore_lang,
                            rescale_with_baseline=True,
                            idf=False
                        )
                        logger.info("Fallback BERTScorer created successfully")
                    except Exception as e2:
                        logger.error(f"Fallback BERTScorer also failed: {e2}")
                        raise e2
            
            scorer = self.bertscorer_cache[cache_key]
            
            # Compute scores for each candidate
            results = []
            for i, (candidate, refs) in enumerate(zip(clean_candidates, clean_references_list)):
                if not candidate.strip() or not refs:
                    logger.warning(f"Empty candidate or references for sample {i+1}, returning zero scores")
                    results.append({"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0})
                    continue
                
                # Use the best reference (first non-empty)
                best_reference = refs[0]
                
                logger.debug(f"Computing BERTScore for sample {i+1}:")
                logger.debug(f"  Candidate: '{candidate}'")
                logger.debug(f"  Reference: '{best_reference}'")
                
                try:
                    P, R, F1 = scorer.score([candidate], [best_reference])
                    
                    result = {
                        "bertscore_precision": float(P.item()),
                        "bertscore_recall": float(R.item()),
                        "bertscore_f1": float(F1.item())
                    }
                    
                    logger.debug(f"Sample {i+1} BERTScore: P={result['bertscore_precision']:.4f}, "
                               f"R={result['bertscore_recall']:.4f}, F1={result['bertscore_f1']:.4f}")
                    
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error computing BERTScore for sample {i+1}: {e}")
                    results.append({"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0})
            
            return results
            
        except Exception as e:
            logger.error(f"Error computing batch BERTScore: {e}")
            return [{"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0} for _ in candidates]
    
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
                           latencies: List[float] = None,
                           language: str = "zh") -> Dict[str, float]:
        """Compute all evaluation metrics including BERTScore with corpus-level IDF"""
        metrics = {}
        
        # Text quality metrics
        bleu_scores = []
        cider_scores = []
        
        # Compute traditional metrics
        for pred, refs in zip(predictions, references):
            bleu = self.compute_bleu(refs, pred)
            cider = self.compute_cider(refs, pred)
            bleu_scores.append(bleu)
            cider_scores.append(cider)
        
        # Compute BERTScore with corpus-level IDF
        bertscore_results = self.compute_batch_bertscore(predictions, references, language=language)
        bertscore_precision_scores = [result['bertscore_precision'] for result in bertscore_results]
        bertscore_recall_scores = [result['bertscore_recall'] for result in bertscore_results]
        bertscore_f1_scores = [result['bertscore_f1'] for result in bertscore_results]
        
        # Average metrics
        metrics['bleu'] = np.mean(bleu_scores)
        metrics['cider'] = np.mean(cider_scores)
        metrics['bertscore_precision'] = np.mean(bertscore_precision_scores)
        metrics['bertscore_recall'] = np.mean(bertscore_recall_scores)
        metrics['bertscore_f1'] = np.mean(bertscore_f1_scores)
        
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
