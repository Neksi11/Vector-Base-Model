"""
Advanced Query Processing Pipeline with intent detection,
query expansion, and semantic analysis.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Set
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

logger = logging.getLogger(__name__)


class QueryProcessor:
    """
    Advanced query processor with intent detection, query expansion,
    and semantic analysis capabilities.
    """
    
    def __init__(self, 
                 enable_spell_correction: bool = True,
                 enable_synonym_expansion: bool = True,
                 max_expanded_terms: int = 5):
        """
        Initialize the query processor.
        
        Args:
            enable_spell_correction: Whether to enable basic spell correction
            enable_synonym_expansion: Whether to expand queries with synonyms
            max_expanded_terms: Maximum number of expanded terms to add
        """
        self.enable_spell_correction = enable_spell_correction
        self.enable_synonym_expansion = enable_synonym_expansion
        self.max_expanded_terms = max_expanded_terms
        
        # Intent patterns
        self.intent_patterns = {
            'definition': [
                r'what is\s+(.+)',
                r'define\s+(.+)',
                r'meaning of\s+(.+)',
                r'(.+)\s+definition'
            ],
            'explanation': [
                r'how does\s+(.+)\s+work',
                r'explain\s+(.+)',
                r'how to\s+(.+)',
                r'why\s+(.+)'
            ],
            'comparison': [
                r'(.+)\s+vs\s+(.+)',
                r'difference between\s+(.+)\s+and\s+(.+)',
                r'compare\s+(.+)\s+and\s+(.+)',
                r'(.+)\s+versus\s+(.+)'
            ],
            'factual': [
                r'when\s+(.+)',
                r'where\s+(.+)',
                r'who\s+(.+)',
                r'which\s+(.+)'
            ],
            'procedural': [
                r'how to\s+(.+)',
                r'steps to\s+(.+)',
                r'process of\s+(.+)',
                r'method for\s+(.+)'
            ]
        }
        
        # Common synonyms for expansion
        self.synonym_dict = {
            'machine learning': ['ml', 'artificial intelligence', 'ai', 'data science'],
            'algorithm': ['method', 'technique', 'approach', 'procedure'],
            'data': ['information', 'dataset', 'records', 'facts'],
            'model': ['framework', 'system', 'structure', 'representation'],
            'analysis': ['examination', 'study', 'investigation', 'evaluation'],
            'classification': ['categorization', 'grouping', 'labeling', 'sorting'],
            'prediction': ['forecast', 'estimation', 'projection', 'anticipation'],
            'performance': ['efficiency', 'effectiveness', 'results', 'outcomes']
        }
        
        # Stop words for cleaning
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 
            'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
        
        logger.info("QueryProcessor initialized")
    
    def process_query(self, 
                     query: str, 
                     expand: bool = True,
                     analyze_intent: bool = True) -> Dict[str, Any]:
        """
        Process a query with comprehensive analysis and expansion.
        
        Args:
            query: Input query string
            expand: Whether to expand the query
            analyze_intent: Whether to analyze query intent
            
        Returns:
            Dictionary containing processed query information
        """
        logger.info(f"Processing query: {query[:50]}...")
        
        # Step 1: Clean and normalize query
        cleaned_query = self._clean_query(query)
        
        # Step 2: Analyze intent
        intent_analysis = {}
        if analyze_intent:
            intent_analysis = self._analyze_intent(cleaned_query)
        
        # Step 3: Extract entities and keywords
        entities = self._extract_entities(cleaned_query)
        keywords = self._extract_keywords(cleaned_query)
        
        # Step 4: Expand query if requested
        expanded_query = cleaned_query
        expansion_terms = []
        if expand:
            expanded_query, expansion_terms = self._expand_query(cleaned_query, keywords)
        
        # Step 5: Generate query variations
        variations = self._generate_query_variations(cleaned_query, entities, keywords)
        
        # Step 6: Calculate query complexity
        complexity_score = self._calculate_complexity(cleaned_query)
        
        return {
            'original_query': query,
            'cleaned_query': cleaned_query,
            'expanded_query': expanded_query,
            'intent': intent_analysis,
            'entities': entities,
            'keywords': keywords,
            'expansion_terms': expansion_terms,
            'variations': variations,
            'complexity_score': complexity_score,
            'metadata': {
                'word_count': len(cleaned_query.split()),
                'char_count': len(cleaned_query),
                'has_question_words': self._has_question_words(cleaned_query),
                'is_complex': complexity_score > 0.6
            }
        }
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize the query."""
        # Convert to lowercase
        cleaned = query.lower().strip()
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove special characters but keep important punctuation
        cleaned = re.sub(r'[^\w\s\?\!\.\,\-]', '', cleaned)
        
        # Basic spell correction (simple replacements)
        if self.enable_spell_correction:
            cleaned = self._basic_spell_correction(cleaned)
        
        return cleaned
    
    def _basic_spell_correction(self, text: str) -> str:
        """Apply basic spell corrections."""
        corrections = {
            'machien': 'machine',
            'learing': 'learning',
            'algoritm': 'algorithm',
            'classifcation': 'classification',
            'prediciton': 'prediction',
            'anaylsis': 'analysis'
        }
        
        for wrong, correct in corrections.items():
            text = text.replace(wrong, correct)
        
        return text
    
    def _analyze_intent(self, query: str) -> Dict[str, Any]:
        """Analyze query intent using pattern matching."""
        detected_intents = []
        intent_confidence = {}
        extracted_entities = {}
        
        for intent_type, patterns in self.intent_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    detected_intents.append(intent_type)
                    intent_confidence[intent_type] = intent_confidence.get(intent_type, 0) + 1
                    
                    # Extract entities from the match
                    if match.groups():
                        extracted_entities[intent_type] = list(match.groups())
        
        # Determine primary intent
        primary_intent = 'general'
        if intent_confidence:
            primary_intent = max(intent_confidence.keys(), key=lambda x: intent_confidence[x])
        
        return {
            'primary_intent': primary_intent,
            'all_intents': detected_intents,
            'confidence_scores': intent_confidence,
            'extracted_entities': extracted_entities
        }
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities and important terms."""
        # Simple entity extraction using capitalization and domain knowledge
        words = query.split()
        entities = []
        
        # Look for capitalized words (potential proper nouns)
        for word in words:
            if word[0].isupper() and len(word) > 2:
                entities.append(word.lower())
        
        # Look for domain-specific terms
        domain_terms = [
            'machine learning', 'deep learning', 'neural network', 'random forest',
            'classification', 'regression', 'clustering', 'supervised learning',
            'unsupervised learning', 'reinforcement learning', 'feature engineering',
            'cross validation', 'overfitting', 'underfitting', 'bias', 'variance'
        ]
        
        query_lower = query.lower()
        for term in domain_terms:
            if term in query_lower:
                entities.append(term)
        
        return list(set(entities))  # Remove duplicates
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from the query."""
        words = query.split()
        
        # Remove stop words and short words
        keywords = [
            word for word in words 
            if word.lower() not in self.stop_words and len(word) > 2
        ]
        
        # Remove common question words
        question_words = {'what', 'how', 'when', 'where', 'why', 'who', 'which'}
        keywords = [word for word in keywords if word.lower() not in question_words]
        
        return keywords
    
    def _expand_query(self, query: str, keywords: List[str]) -> tuple[str, List[str]]:
        """Expand query with synonyms and related terms."""
        if not self.enable_synonym_expansion:
            return query, []
        
        expansion_terms = []
        query_lower = query.lower()
        
        # Add synonyms for keywords
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # Check direct synonyms
            if keyword_lower in self.synonym_dict:
                synonyms = self.synonym_dict[keyword_lower]
                expansion_terms.extend(synonyms[:2])  # Limit to 2 synonyms per keyword
            
            # Check if keyword is part of a multi-word term
            for term, synonyms in self.synonym_dict.items():
                if keyword_lower in term and term in query_lower:
                    expansion_terms.extend(synonyms[:2])
        
        # Limit total expansion terms
        expansion_terms = expansion_terms[:self.max_expanded_terms]
        
        # Create expanded query
        if expansion_terms:
            expanded_query = query + " " + " ".join(expansion_terms)
        else:
            expanded_query = query
        
        return expanded_query, expansion_terms
    
    def _generate_query_variations(self, 
                                  query: str, 
                                  entities: List[str], 
                                  keywords: List[str]) -> List[str]:
        """Generate variations of the query for better retrieval."""
        variations = [query]  # Include original
        
        # Keyword-only variation
        if keywords:
            keyword_query = " ".join(keywords)
            if keyword_query != query:
                variations.append(keyword_query)
        
        # Entity-focused variation
        if entities:
            entity_query = " ".join(entities)
            if entity_query != query and entity_query not in variations:
                variations.append(entity_query)
        
        # Combined entities and keywords
        if entities and keywords:
            combined = list(set(entities + keywords))
            combined_query = " ".join(combined)
            if combined_query not in variations:
                variations.append(combined_query)
        
        return variations[:5]  # Limit to 5 variations
    
    def _calculate_complexity(self, query: str) -> float:
        """Calculate query complexity score (0.0 to 1.0)."""
        factors = []
        
        # Length factor
        word_count = len(query.split())
        length_factor = min(word_count / 20.0, 1.0)  # Normalize to 20 words
        factors.append(length_factor)
        
        # Question word factor
        question_words = ['what', 'how', 'when', 'where', 'why', 'who', 'which']
        question_factor = sum(1 for word in query.split() if word.lower() in question_words) / len(query.split())
        factors.append(question_factor)
        
        # Technical term factor
        technical_terms = ['algorithm', 'model', 'classification', 'regression', 'neural', 'machine learning']
        tech_factor = sum(1 for term in technical_terms if term in query.lower()) / max(len(technical_terms), 1)
        factors.append(tech_factor)
        
        # Conjunction factor (complex queries often have multiple clauses)
        conjunctions = ['and', 'or', 'but', 'however', 'while', 'whereas']
        conj_factor = sum(1 for conj in conjunctions if conj in query.lower()) / max(len(conjunctions), 1)
        factors.append(conj_factor)
        
        # Average the factors
        return np.mean(factors)
    
    def _has_question_words(self, query: str) -> bool:
        """Check if query contains question words."""
        question_words = ['what', 'how', 'when', 'where', 'why', 'who', 'which']
        return any(word in query.lower().split() for word in question_words)
