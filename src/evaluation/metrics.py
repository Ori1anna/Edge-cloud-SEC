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

# METEOR imports
try:
    from nltk.translate.meteor_score import meteor_score
    METEOR_AVAILABLE = True
    logger.info("METEOR is available")
except ImportError:
    METEOR_AVAILABLE = False
    logger.warning("METEOR not available. Install with: pip install nltk")

# ROUGE imports
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
    logger.info("ROUGE is available")
except ImportError:
    ROUGE_AVAILABLE = False
    logger.warning("ROUGE not available. Install with: pip install rouge-score")

# PyCOCOEvalCap imports
try:
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.cider.cider import Cider
    PYCOCOEVALCAP_AVAILABLE = True
    logger.info("PyCOCOEvalCap is available")
except ImportError:
    PYCOCOEVALCAP_AVAILABLE = False
    logger.warning("PyCOCOEvalCap not available. Install with: pip install pycocoevalcap")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')


class EvaluationMetrics:
    """Evaluation metrics for speech emotion captioning"""
    
    def __init__(self):
        self.smoothing = SmoothingFunction().method1
        self.bertscorer_cache = {}  # Cache for BERTScorer objects
    
    def compute_bleu(self, references: List[str], hypothesis: str, language: str = "chinese", n_gram: int = 4) -> float:
        """Compute BLEU-n score with language-aware tokenization"""
        try:
            # Clean hypothesis text (remove special tokens)
            hypothesis = hypothesis.replace('<|im_end|>', '').strip()
            
            # 根据语言选择分词方式
            if language.lower() in ["english", "en"]:
                # 英文使用词语级分词
                ref_tokens = [word_tokenize(ref.lower()) for ref in references]
                hyp_tokens = word_tokenize(hypothesis.lower())
            else:
                # 中文或其他语言使用字符级分词
                ref_tokens = [list(ref.lower()) for ref in references]
                hyp_tokens = list(hypothesis.lower())

            logger.debug(f"Language: {language}, Tokenization: {'word' if language.lower() in ['english', 'en'] else 'character'}")
            logger.debug(f"Reference tokens: {ref_tokens}")
            logger.debug(f"Hypothesis tokens: {hyp_tokens}")
            
            # Compute BLEU with specified n-gram
            weights = [1/n_gram] * n_gram + [0] * (4 - n_gram)  # Only use first n_gram weights
            bleu_score = sentence_bleu(ref_tokens, hyp_tokens, weights=weights, smoothing_function=self.smoothing)
            logger.debug(f"BLEU-{n_gram} score: {bleu_score}")
            
            return bleu_score
        except Exception as e:
            logger.error(f"Error computing BLEU-{n_gram}: {e}")
            return 0.0
    
    def compute_bleu_1(self, references: List[str], hypothesis: str, language: str = "chinese") -> float:
        """Compute BLEU-1 score"""
        return self.compute_bleu(references, hypothesis, language=language, n_gram=1)
    
    def compute_bleu_4(self, references: List[str], hypothesis: str, language: str = "chinese") -> float:
        """Compute BLEU-4 score"""
        return self.compute_bleu(references, hypothesis, language=language, n_gram=4)
    
    def compute_cider(self, references: List[str], hypothesis: str, language: str = "chinese") -> float:
        """Compute CIDEr score - DISABLED, returns 0.0"""
        # CIDEr calculation disabled per user request
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
    
    def compute_meteor(self, references: List[str], hypothesis: str, language: str = "chinese") -> float:
        """
        Compute METEOR score
        
        Args:
            references: List of reference texts
            hypothesis: Generated text
            language: Language parameter ("chinese", "english", etc.)
            
        Returns:
            METEOR score
        """
        # Use pycocoevalcap METEOR if available and Java is installed
        use_pycocoevalcap = PYCOCOEVALCAP_AVAILABLE
        
        if not use_pycocoevalcap or not PYCOCOEVALCAP_AVAILABLE:
            logger.warning("PyCOCOEvalCap not available, falling back to NLTK METEOR")
            if not METEOR_AVAILABLE:
                logger.warning("METEOR not available, returning zero score")
                return 0.0
            
            try:
                # Clean texts
                hypothesis = hypothesis.replace('<|im_end|>', '').strip()
                clean_references = [ref.replace('<|im_end|>', '').strip() for ref in references]
                
                # Skip empty texts
                if not hypothesis.strip() or not any(ref.strip() for ref in clean_references):
                    logger.warning("Empty hypothesis or references, returning zero METEOR score")
                    return 0.0
                
                # Use the best reference (first non-empty)
                best_reference = None
                for ref in clean_references:
                    if ref.strip():
                        best_reference = ref.strip()
                        break
                
                if not best_reference:
                    logger.warning("No valid reference found, returning zero METEOR score")
                    return 0.0
                
                # Tokenization based on language
                if language.lower() in ["english", "en"]:
                    # Use word tokenization for English
                    ref_tokens = word_tokenize(best_reference.lower())
                    hyp_tokens = word_tokenize(hypothesis.lower())
                else:
                    # Use character tokenization for Chinese and other languages
                    ref_tokens = list(best_reference.lower())
                    hyp_tokens = list(hypothesis.lower())
                
                logger.debug(f"Computing METEOR for hypothesis: '{hypothesis[:50]}...'")
                logger.debug(f"Computing METEOR for reference: '{best_reference[:50]}...'")
                logger.debug(f"Language: {language}, Tokenization: {'word' if language.lower() in ['english', 'en'] else 'character'}")
                
                # Compute METEOR score
                meteor_score_value = meteor_score([ref_tokens], hyp_tokens)
                
                # Handle potential NaN or None values
                if meteor_score_value is None or str(meteor_score_value).lower() == 'nan':
                    logger.warning(f"METEOR returned invalid value, using 0.0")
                    meteor_score_value = 0.0
                
                logger.debug(f"METEOR score: {meteor_score_value}")
                return float(meteor_score_value)
                
            except Exception as e:
                logger.error(f"Error computing METEOR: {e}")
                return 0.0
        if use_pycocoevalcap:
            try:
                # Clean texts
                hypothesis = hypothesis.replace('<|im_end|>', '').strip()
                clean_references = [ref.replace('<|im_end|>', '').strip() for ref in references]
                
                # Skip empty texts
                if not hypothesis.strip() or not any(ref.strip() for ref in clean_references):
                    logger.warning("Empty hypothesis or references, returning zero METEOR score")
                    return 0.0
                
                # Use the best reference (first non-empty)
                best_reference = None
                for ref in clean_references:
                    if ref.strip():
                        best_reference = ref.strip()
                        break
                
                if not best_reference:
                    logger.warning("No valid reference found, returning zero METEOR score")
                    return 0.0
                
                logger.debug(f"Computing METEOR for hypothesis: '{hypothesis[:50]}...'")
                logger.debug(f"Computing METEOR for reference: '{best_reference[:50]}...'")
                logger.debug(f"Language: {language}")
                
                # Initialize Meteor scorer
                meteor_scorer = Meteor()
                
                # Format for PyCOCOEvalCap: dictionaries with index keys
                # PyCOCOEvalCap expects strings (not token lists)
                gts = {0: [best_reference]}
                res = {0: [hypothesis]}
                
                # Compute METEOR score
                meteor_score_value, _ = meteor_scorer.compute_score(gts, res)
                
                # Handle potential NaN or None values
                if meteor_score_value is None or str(meteor_score_value).lower() == 'nan':
                    logger.warning(f"METEOR returned invalid value, using 0.0")
                    meteor_score_value = 0.0
                
                logger.debug(f"METEOR score: {meteor_score_value}")
                return float(meteor_score_value)
                
            except Exception as e:
                logger.error(f"Error computing METEOR: {e}")
                return 0.0
    
    def compute_rouge_l(self, references: List[str], hypothesis: str) -> float:
        """
        Compute ROUGE-L score
        
        Args:
            references: List of reference texts
            hypothesis: Generated text
            
        Returns:
            ROUGE-L F1 score
        """
        if not ROUGE_AVAILABLE:
            logger.warning("ROUGE not available, returning zero score")
            return 0.0
        
        try:
            # Clean texts
            hypothesis = hypothesis.replace('<|im_end|>', '').strip()
            clean_references = [ref.replace('<|im_end|>', '').strip() for ref in references]
            
            # Skip empty texts
            if not hypothesis.strip() or not any(ref.strip() for ref in clean_references):
                logger.warning("Empty hypothesis or references, returning zero ROUGE-L score")
                return 0.0
            
            # Use the best reference (first non-empty)
            best_reference = None
            for ref in clean_references:
                if ref.strip():
                    best_reference = ref.strip()
                    break
            
            if not best_reference:
                logger.warning("No valid reference found, returning zero ROUGE-L score")
                return 0.0
            
            logger.debug(f"Computing ROUGE-L for hypothesis: '{hypothesis[:50]}...'")
            logger.debug(f"Computing ROUGE-L for reference: '{best_reference[:50]}...'")
            
            # Initialize ROUGE scorer
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            
            # Compute ROUGE-L score
            scores = scorer.score(best_reference, hypothesis)
            
            # Handle case where ROUGE-L might return 0.0
            if scores and 'rougeL' in scores:
                rouge_l_f1 = scores['rougeL'].fmeasure
                # Ensure we don't return NaN or None
                if rouge_l_f1 is None or str(rouge_l_f1).lower() == 'nan':
                    rouge_l_f1 = 0.0
                logger.debug(f"ROUGE-L F1 score: {rouge_l_f1}")
                return float(rouge_l_f1)
            else:
                logger.warning(f"ROUGE-L calculation failed for hypothesis: '{hypothesis[:50]}...'")
                return 0.0
            
        except Exception as e:
            logger.error(f"Error computing ROUGE-L: {e}")
            return 0.0
    
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
                rescale_with_baseline=False,  # CRITICAL: Disable rescaling to prevent negative scores
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
                        rescale_with_baseline=True,  # CRITICAL: Disable rescaling to prevent negative scores
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
                            rescale_with_baseline=False,  # Disable rescaling to prevent negative scores
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
    
    def compute_corpus_meteor(self, hypotheses: List[str], references: List[str], language: str = "chinese") -> float:
        """
        Compute corpus-level METEOR score using pycocoevalcap
        
        Args:
            hypotheses: List of generated texts
            references: List of reference texts
            language: Language parameter ("chinese", "english", etc.)
            
        Returns:
            Corpus-level METEOR score
        """
        # Use pycocoevalcap METEOR if available, otherwise fallback to sentence-level average
        if not PYCOCOEVALCAP_AVAILABLE:
            logger.warning("PyCOCOEvalCap not available for corpus METEOR, using average sentence-level score")
            meteor_scores = []
            for hyp, ref in zip(hypotheses, references):
                meteor_scores.append(self.compute_meteor([ref], hyp, language=language))
            return np.mean(meteor_scores) if meteor_scores else 0.0
        
        try:
            if not hypotheses or not references or len(hypotheses) != len(references):
                logger.error("Invalid input for corpus METEOR computation")
                return 0.0
            
            # Clean texts: remove special tokens, newlines, tabs, and strip
            # This is critical for pycocoevalcap METEOR which writes to temp files
            # Unhandled newlines/tabs can break Java METEOR's file parsing
            cleaned_references = [ref.replace('<|im_end|>', '').replace('\n', ' ').replace('\t', ' ').strip() for ref in references]
            cleaned_hypotheses = [hyp.replace('<|im_end|>', '').replace('\n', ' ').replace('\t', ' ').strip() for hyp in hypotheses]
            
            # Format for PyCOCOEvalCap: dictionaries with index keys
            # PyCOCOEvalCap expects strings (not token lists)
            gts = {i: [ref] for i, ref in enumerate(cleaned_references)}
            res = {i: [hyp] for i, hyp in enumerate(cleaned_hypotheses)}
            
            # Initialize Meteor scorer
            meteor_scorer = Meteor()
            
            # Compute corpus-level METEOR score
            meteor_score_value, _ = meteor_scorer.compute_score(gts, res)
            
            # Handle potential NaN or None values
            if meteor_score_value is None or str(meteor_score_value).lower() == 'nan':
                logger.warning(f"Corpus METEOR returned invalid value, using 0.0")
                meteor_score_value = 0.0
            
            logger.debug(f"Corpus METEOR score: {meteor_score_value}")
            return float(meteor_score_value)
            
        except Exception as e:
            logger.error(f"Error computing corpus METEOR: {e}")
            # Fallback to sentence-level average
            try:
                meteor_scores = []
                for hyp, ref in zip(hypotheses, references):
                    meteor_scores.append(self.compute_meteor([ref], hyp, language=language))
                logger.warning("Falling back to average sentence-level METEOR")
                return np.mean(meteor_scores) if meteor_scores else 0.0
            except:
                return 0.0
    
    def compute_corpus_rouge_l(self, hypotheses: List[str], references: List[str]) -> float:
        """
        Compute corpus-level ROUGE-L score
        
        Args:
            hypotheses: List of generated texts
            references: List of reference texts
            
        Returns:
            Corpus-level ROUGE-L F1 score
        """
        if not ROUGE_AVAILABLE:
            logger.warning("ROUGE not available for corpus ROUGE-L, returning average sentence-level score")
            # Fallback to average sentence-level
            rouge_l_scores = []
            for hyp, ref in zip(hypotheses, references):
                rouge_l_scores.append(self.compute_rouge_l([ref], hyp))
            return np.mean(rouge_l_scores) if rouge_l_scores else 0.0
        
        try:
            if not hypotheses or not references or len(hypotheses) != len(references):
                logger.error("Invalid input for corpus ROUGE-L computation")
                return 0.0
            
            # Initialize ROUGE scorer
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            
            # Compute ROUGE-L for each pair
            rouge_scores = []
            for hyp, ref in zip(hypotheses, references):
                scores = scorer.score(ref, hyp)
                rouge_l_f1 = scores['rougeL'].fmeasure
                if rouge_l_f1 is not None and str(rouge_l_f1).lower() != 'nan':
                    rouge_scores.append(float(rouge_l_f1))
            
            # Average across all samples for corpus-level score
            corpus_score = np.mean(rouge_scores) if rouge_scores else 0.0
            logger.debug(f"Corpus ROUGE-L score: {corpus_score}")
            return corpus_score
            
        except Exception as e:
            logger.error(f"Error computing corpus ROUGE-L: {e}")
            return 0.0
    
    def compute_corpus_cider(self, hypotheses: List[str], references: List[str], language: str = "chinese") -> float:
        """
        Compute corpus-level CIDEr score - DISABLED, returns 0.0
        
        Args:
            hypotheses: List of generated texts
            references: List of reference texts
            language: Language parameter ("chinese", "english", etc.)
            
        Returns:
            Corpus-level CIDEr score (always 0.0)
        """
        # CIDEr calculation disabled per user request
        return 0.0

    def compute_all_metrics(self, 
                           predictions: List[str],
                           references: List[List[str]],
                           predicted_emotions: List[str] = None,
                           reference_emotions: List[str] = None,
                           latencies: List[float] = None,
                           language: str = "zh") -> Dict[str, float]:
        """Compute all evaluation metrics including BLEU-1, BLEU-4, METEOR, ROUGE-L, CIDEr, and BERTScore"""
        metrics = {}
        
        # Text quality metrics
        bleu_1_scores = []
        bleu_4_scores = []
        meteor_scores = []
        rouge_l_scores = []
        cider_scores = []
        
        # Compute traditional metrics
        for pred, refs in zip(predictions, references):
            bleu_1 = self.compute_bleu_1(refs, pred, language=language)
            bleu_4 = self.compute_bleu_4(refs, pred, language=language)
            meteor = self.compute_meteor(refs, pred, language=language)
            rouge_l = self.compute_rouge_l(refs, pred)
            cider = self.compute_cider(refs, pred, language=language)
            
            bleu_1_scores.append(bleu_1)
            bleu_4_scores.append(bleu_4)
            meteor_scores.append(meteor)
            rouge_l_scores.append(rouge_l)
            cider_scores.append(cider)
        
        # Compute BERTScore with corpus-level IDF
        bertscore_results = self.compute_batch_bertscore(predictions, references, language=language)
        bertscore_precision_scores = [result['bertscore_precision'] for result in bertscore_results]
        bertscore_recall_scores = [result['bertscore_recall'] for result in bertscore_results]
        bertscore_f1_scores = [result['bertscore_f1'] for result in bertscore_results]
        
        # Average metrics
        metrics['bleu_1'] = np.mean(bleu_1_scores)
        metrics['bleu_4'] = np.mean(bleu_4_scores)
        metrics['meteor'] = np.mean(meteor_scores)
        metrics['rouge_l'] = np.mean(rouge_l_scores)
        metrics['cider'] = np.mean(cider_scores)
        metrics['bertscore_precision'] = np.mean(bertscore_precision_scores)
        metrics['bertscore_recall'] = np.mean(bertscore_recall_scores)
        metrics['bertscore_f1'] = np.mean(bertscore_f1_scores)
        
        # Backward compatibility: keep old 'bleu' key for existing code
        metrics['bleu'] = metrics['bleu_4']
        
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
