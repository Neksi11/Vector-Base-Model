"""
Comprehensive Evaluation and Metrics System for RAG performance assessment.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Comprehensive evaluator for RAG system performance including
    retrieval quality, response relevance, and system metrics.
    """
    
    def __init__(self):
        """Initialize the RAG evaluator."""
        self.evaluation_history = []
        self.performance_metrics = {
            'total_queries': 0,
            'avg_response_time': 0.0,
            'avg_retrieval_score': 0.0,
            'avg_response_relevance': 0.0,
            'avg_confidence': 0.0
        }
        
        logger.info("RAGEvaluator initialized")
    
    def evaluate_response(self, 
                         query: str,
                         response: str,
                         retrieved_docs: List[Dict[str, Any]],
                         ground_truth: Optional[str] = None,
                         start_time: Optional[float] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a RAG response.
        
        Args:
            query: Original user query
            response: Generated response
            retrieved_docs: Documents retrieved for the query
            ground_truth: Optional ground truth answer for comparison
            start_time: Optional start time for response time calculation
            
        Returns:
            Dictionary containing comprehensive evaluation metrics
        """
        evaluation_start = time.time()
        
        # Calculate response time if start_time provided
        response_time = time.time() - start_time if start_time else None
        
        # Evaluate different aspects
        retrieval_metrics = self._evaluate_retrieval(query, retrieved_docs)
        relevance_metrics = self._evaluate_relevance(query, response, retrieved_docs)
        quality_metrics = self._evaluate_response_quality(response)
        
        # Ground truth comparison if available
        ground_truth_metrics = {}
        if ground_truth:
            ground_truth_metrics = self._evaluate_against_ground_truth(response, ground_truth)
        
        # Compile comprehensive evaluation
        evaluation = {
            'retrieval_metrics': retrieval_metrics,
            'relevance_metrics': relevance_metrics,
            'quality_metrics': quality_metrics,
            'ground_truth_metrics': ground_truth_metrics,
            'system_metrics': {
                'response_time': response_time,
                'evaluation_time': time.time() - evaluation_start,
                'num_retrieved_docs': len(retrieved_docs)
            },
            'overall_score': self._calculate_overall_score(
                retrieval_metrics, relevance_metrics, quality_metrics, ground_truth_metrics
            )
        }
        
        # Update performance tracking
        self._update_performance_metrics(evaluation)
        
        # Store in history
        self.evaluation_history.append({
            'query': query,
            'response': response,
            'evaluation': evaluation,
            'timestamp': time.time()
        })
        
        return evaluation
    
    def _evaluate_retrieval(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate retrieval quality metrics."""
        if not retrieved_docs:
            return {
                'num_documents': 0,
                'avg_score': 0.0,
                'score_variance': 0.0,
                'coverage_score': 0.0,
                'diversity_score': 0.0
            }
        
        scores = [doc.get('score', 0.0) for doc in retrieved_docs]
        
        # Basic statistics
        avg_score = np.mean(scores)
        score_variance = np.var(scores)
        
        # Coverage score (how well documents cover the query)
        coverage_score = self._calculate_coverage_score(query, retrieved_docs)
        
        # Diversity score (how diverse the retrieved documents are)
        diversity_score = self._calculate_diversity_score(retrieved_docs)
        
        return {
            'num_documents': len(retrieved_docs),
            'avg_score': avg_score,
            'max_score': max(scores),
            'min_score': min(scores),
            'score_variance': score_variance,
            'coverage_score': coverage_score,
            'diversity_score': diversity_score,
            'score_distribution': self._analyze_score_distribution(scores)
        }
    
    def _evaluate_relevance(self, 
                           query: str, 
                           response: str, 
                           retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate response relevance to query and retrieved documents."""
        # Query-response relevance
        query_relevance = self._calculate_text_similarity(query, response)
        
        # Response-documents relevance
        doc_texts = [doc['document'] for doc in retrieved_docs]
        doc_relevance_scores = []
        
        if doc_texts:
            for doc_text in doc_texts:
                relevance = self._calculate_text_similarity(response, doc_text)
                doc_relevance_scores.append(relevance)
        
        avg_doc_relevance = np.mean(doc_relevance_scores) if doc_relevance_scores else 0.0
        
        # Context utilization (how well the response uses retrieved context)
        context_utilization = self._calculate_context_utilization(response, doc_texts)
        
        return {
            'query_relevance': query_relevance,
            'avg_document_relevance': avg_doc_relevance,
            'max_document_relevance': max(doc_relevance_scores) if doc_relevance_scores else 0.0,
            'context_utilization': context_utilization,
            'relevance_consistency': np.std(doc_relevance_scores) if doc_relevance_scores else 0.0
        }
    
    def _evaluate_response_quality(self, response: str) -> Dict[str, Any]:
        """Evaluate intrinsic response quality metrics."""
        # Length metrics
        word_count = len(response.split())
        char_count = len(response)
        
        # Readability metrics
        readability_score = self._calculate_readability(response)
        
        # Coherence metrics
        coherence_score = self._calculate_coherence(response)
        
        # Completeness metrics
        completeness_score = self._calculate_completeness(response)
        
        # Factual consistency (basic checks)
        consistency_score = self._calculate_consistency(response)
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'readability_score': readability_score,
            'coherence_score': coherence_score,
            'completeness_score': completeness_score,
            'consistency_score': consistency_score,
            'has_proper_structure': self._has_proper_structure(response)
        }
    
    def _evaluate_against_ground_truth(self, response: str, ground_truth: str) -> Dict[str, Any]:
        """Evaluate response against ground truth answer."""
        # Semantic similarity
        semantic_similarity = self._calculate_text_similarity(response, ground_truth)
        
        # BLEU-like score (simple n-gram overlap)
        bleu_score = self._calculate_bleu_score(response, ground_truth)
        
        # Factual overlap
        factual_overlap = self._calculate_factual_overlap(response, ground_truth)
        
        return {
            'semantic_similarity': semantic_similarity,
            'bleu_score': bleu_score,
            'factual_overlap': factual_overlap,
            'length_ratio': len(response) / len(ground_truth) if ground_truth else 0.0
        }
    
    def _calculate_coverage_score(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> float:
        """Calculate how well retrieved documents cover the query."""
        if not retrieved_docs:
            return 0.0
        
        query_terms = set(query.lower().split())
        covered_terms = set()
        
        for doc in retrieved_docs:
            doc_terms = set(doc['document'].lower().split())
            covered_terms.update(query_terms.intersection(doc_terms))
        
        coverage = len(covered_terms) / len(query_terms) if query_terms else 0.0
        return coverage
    
    def _calculate_diversity_score(self, retrieved_docs: List[Dict[str, Any]]) -> float:
        """Calculate diversity among retrieved documents."""
        if len(retrieved_docs) < 2:
            return 1.0  # Single document is perfectly diverse
        
        doc_texts = [doc['document'] for doc in retrieved_docs]
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(doc_texts)):
            for j in range(i + 1, len(doc_texts)):
                sim = self._calculate_text_similarity(doc_texts[i], doc_texts[j])
                similarities.append(sim)
        
        # Diversity is inverse of average similarity
        avg_similarity = np.mean(similarities) if similarities else 0.0
        diversity = 1.0 - avg_similarity
        
        return max(0.0, diversity)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            return 0.0
    
    def _calculate_context_utilization(self, response: str, doc_texts: List[str]) -> float:
        """Calculate how well the response utilizes the retrieved context."""
        if not doc_texts:
            return 0.0
        
        response_words = set(response.lower().split())
        context_words = set()
        
        for doc_text in doc_texts:
            context_words.update(doc_text.lower().split())
        
        # Calculate overlap
        overlap = len(response_words.intersection(context_words))
        utilization = overlap / len(response_words) if response_words else 0.0
        
        return min(utilization, 1.0)
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score (simplified)."""
        sentences = len(re.split(r'[.!?]+', text))
        words = len(text.split())
        
        if sentences == 0:
            return 0.0
        
        avg_sentence_length = words / sentences
        
        # Simple readability score (inverse of average sentence length, normalized)
        readability = 1.0 / (1.0 + avg_sentence_length / 20.0)
        
        return readability
    
    def _calculate_coherence(self, text: str) -> float:
        """Calculate coherence score based on sentence transitions."""
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        if len(sentences) < 2:
            return 1.0
        
        # Calculate similarity between consecutive sentences
        coherence_scores = []
        for i in range(len(sentences) - 1):
            similarity = self._calculate_text_similarity(sentences[i], sentences[i + 1])
            coherence_scores.append(similarity)
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    def _calculate_completeness(self, text: str) -> float:
        """Calculate completeness score based on structure and content."""
        # Check for proper ending
        has_proper_ending = text.strip().endswith(('.', '!', '?'))
        
        # Check for reasonable length
        word_count = len(text.split())
        length_score = min(word_count / 50.0, 1.0)  # Normalize to 50 words
        
        # Check for structure indicators
        has_structure = any(indicator in text.lower() for indicator in 
                          ['first', 'second', 'finally', 'in conclusion', 'therefore', 'however'])
        
        completeness = (
            0.4 * (1.0 if has_proper_ending else 0.0) +
            0.4 * length_score +
            0.2 * (1.0 if has_structure else 0.0)
        )
        
        return completeness
    
    def _calculate_consistency(self, text: str) -> float:
        """Calculate internal consistency score."""
        # Simple consistency check based on contradictory terms
        contradictory_pairs = [
            ('always', 'never'), ('all', 'none'), ('increase', 'decrease'),
            ('positive', 'negative'), ('true', 'false'), ('yes', 'no')
        ]
        
        text_lower = text.lower()
        contradictions = 0
        
        for term1, term2 in contradictory_pairs:
            if term1 in text_lower and term2 in text_lower:
                contradictions += 1
        
        # Consistency decreases with contradictions
        consistency = 1.0 / (1.0 + contradictions)
        
        return consistency
    
    def _has_proper_structure(self, text: str) -> bool:
        """Check if response has proper structure."""
        # Check for proper capitalization
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        if not sentences:
            return False
        
        # Check if sentences start with capital letters
        proper_capitalization = all(s[0].isupper() for s in sentences if s)
        
        # Check for proper punctuation
        proper_punctuation = text.strip().endswith(('.', '!', '?'))
        
        return proper_capitalization and proper_punctuation

    def _analyze_score_distribution(self, scores: List[float]) -> Dict[str, float]:
        """Analyze the distribution of retrieval scores."""
        if not scores:
            return {'mean': 0.0, 'std': 0.0, 'skewness': 0.0}

        mean_score = np.mean(scores)
        std_score = np.std(scores)

        # Simple skewness calculation
        if std_score > 0:
            skewness = np.mean([(score - mean_score) ** 3 for score in scores]) / (std_score ** 3)
        else:
            skewness = 0.0

        return {
            'mean': mean_score,
            'std': std_score,
            'skewness': skewness
        }

    def _calculate_bleu_score(self, response: str, ground_truth: str) -> float:
        """Calculate a simplified BLEU-like score."""
        response_words = response.lower().split()
        truth_words = ground_truth.lower().split()

        if not response_words or not truth_words:
            return 0.0

        # Calculate 1-gram precision
        response_set = set(response_words)
        truth_set = set(truth_words)

        overlap = len(response_set.intersection(truth_set))
        precision = overlap / len(response_set) if response_set else 0.0

        return precision

    def _calculate_factual_overlap(self, response: str, ground_truth: str) -> float:
        """Calculate factual overlap between response and ground truth."""
        # Extract potential facts (simple approach using noun phrases)
        response_facts = self._extract_facts(response)
        truth_facts = self._extract_facts(ground_truth)

        if not response_facts or not truth_facts:
            return 0.0

        overlap = len(set(response_facts).intersection(set(truth_facts)))
        total_facts = len(set(response_facts).union(set(truth_facts)))

        return overlap / total_facts if total_facts > 0 else 0.0

    def _extract_facts(self, text: str) -> List[str]:
        """Extract potential facts from text (simplified approach)."""
        # Simple fact extraction using patterns
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        facts = []

        for sentence in sentences:
            # Look for factual patterns
            if any(pattern in sentence.lower() for pattern in ['is', 'are', 'was', 'were', 'has', 'have']):
                facts.append(sentence.lower())

        return facts

    def _calculate_overall_score(self,
                               retrieval_metrics: Dict[str, Any],
                               relevance_metrics: Dict[str, Any],
                               quality_metrics: Dict[str, Any],
                               ground_truth_metrics: Dict[str, Any]) -> float:
        """Calculate overall performance score."""
        scores = []

        # Retrieval score (30% weight)
        retrieval_score = retrieval_metrics.get('avg_score', 0.0)
        scores.append(0.3 * retrieval_score)

        # Relevance score (40% weight)
        relevance_score = (
            relevance_metrics.get('query_relevance', 0.0) * 0.6 +
            relevance_metrics.get('avg_document_relevance', 0.0) * 0.4
        )
        scores.append(0.4 * relevance_score)

        # Quality score (20% weight)
        quality_score = (
            quality_metrics.get('coherence_score', 0.0) * 0.4 +
            quality_metrics.get('completeness_score', 0.0) * 0.3 +
            quality_metrics.get('consistency_score', 0.0) * 0.3
        )
        scores.append(0.2 * quality_score)

        # Ground truth score (10% weight, if available)
        if ground_truth_metrics:
            gt_score = ground_truth_metrics.get('semantic_similarity', 0.0)
            scores.append(0.1 * gt_score)

        return sum(scores)

    def _update_performance_metrics(self, evaluation: Dict[str, Any]):
        """Update running performance metrics."""
        self.performance_metrics['total_queries'] += 1

        # Update averages
        total = self.performance_metrics['total_queries']

        # Response time
        response_time = evaluation['system_metrics'].get('response_time')
        if response_time:
            current_avg = self.performance_metrics['avg_response_time']
            self.performance_metrics['avg_response_time'] = (
                (current_avg * (total - 1) + response_time) / total
            )

        # Retrieval score
        retrieval_score = evaluation['retrieval_metrics'].get('avg_score', 0.0)
        current_avg = self.performance_metrics['avg_retrieval_score']
        self.performance_metrics['avg_retrieval_score'] = (
            (current_avg * (total - 1) + retrieval_score) / total
        )

        # Response relevance
        relevance_score = evaluation['relevance_metrics'].get('query_relevance', 0.0)
        current_avg = self.performance_metrics['avg_response_relevance']
        self.performance_metrics['avg_response_relevance'] = (
            (current_avg * (total - 1) + relevance_score) / total
        )

        # Overall confidence
        overall_score = evaluation.get('overall_score', 0.0)
        current_avg = self.performance_metrics['avg_confidence']
        self.performance_metrics['avg_confidence'] = (
            (current_avg * (total - 1) + overall_score) / total
        )

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            'current_metrics': self.performance_metrics.copy(),
            'evaluation_history_length': len(self.evaluation_history),
            'recent_performance': self._get_recent_performance(),
            'performance_trends': self._analyze_performance_trends()
        }

    def _get_recent_performance(self, window_size: int = 10) -> Dict[str, Any]:
        """Get performance metrics for recent evaluations."""
        if len(self.evaluation_history) < window_size:
            recent_evals = self.evaluation_history
        else:
            recent_evals = self.evaluation_history[-window_size:]

        if not recent_evals:
            return {}

        recent_scores = [eval_data['evaluation']['overall_score'] for eval_data in recent_evals]
        recent_retrieval = [eval_data['evaluation']['retrieval_metrics']['avg_score']
                          for eval_data in recent_evals]

        return {
            'avg_overall_score': np.mean(recent_scores),
            'avg_retrieval_score': np.mean(recent_retrieval),
            'score_trend': 'improving' if len(recent_scores) > 1 and recent_scores[-1] > recent_scores[0] else 'stable'
        }

    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        if len(self.evaluation_history) < 5:
            return {'trend': 'insufficient_data'}

        # Get overall scores over time
        scores = [eval_data['evaluation']['overall_score'] for eval_data in self.evaluation_history]

        # Simple trend analysis
        recent_avg = np.mean(scores[-5:])
        earlier_avg = np.mean(scores[:-5])

        if recent_avg > earlier_avg + 0.05:
            trend = 'improving'
        elif recent_avg < earlier_avg - 0.05:
            trend = 'declining'
        else:
            trend = 'stable'

        return {
            'trend': trend,
            'recent_avg': recent_avg,
            'earlier_avg': earlier_avg,
            'improvement': recent_avg - earlier_avg
        }
