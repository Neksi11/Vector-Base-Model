"""
Sophisticated Response Generator with template-based generation,
context fusion, and answer synthesis capabilities.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """
    Advanced response generator with multiple generation strategies,
    context fusion, and answer synthesis.
    """
    
    def __init__(self, 
                 max_length: int = 4000,
                 temperature: float = 0.7,
                 response_templates: Optional[Dict[str, str]] = None):
        """
        Initialize the response generator.
        
        Args:
            max_length: Maximum response length
            temperature: Generation temperature (0.0 to 1.0)
            response_templates: Custom response templates
        """
        self.max_length = max_length
        self.temperature = temperature
        
        # Default response templates
        self.response_templates = response_templates or {
            'factual': "Based on the available information: {context}\n\nAnswer: {answer}",
            'analytical': "Analysis of the query '{query}':\n\n{context}\n\nConclusion: {answer}",
            'comparative': "Comparing the available information:\n\n{context}\n\nSummary: {answer}",
            'explanatory': "To explain '{query}':\n\n{context}\n\nExplanation: {answer}",
            'default': "Query: {query}\n\nRelevant Information:\n{context}\n\nResponse: {answer}"
        }
        
        # Context fusion settings
        self.context_fusion_method = 'weighted_combination'
        self.max_context_chunks = 5
        
        logger.info("ResponseGenerator initialized")
    
    def generate_response(self, 
                         query: str,
                         context_documents: List[Dict[str, Any]],
                         max_context_length: Optional[int] = None,
                         response_type: str = 'default') -> Dict[str, Any]:
        """
        Generate a comprehensive response using advanced techniques.
        
        Args:
            query: User query
            context_documents: Retrieved context documents
            max_context_length: Maximum context length to use
            response_type: Type of response template to use
            
        Returns:
            Dictionary containing response and metadata
        """
        logger.info(f"Generating response for query: {query[:50]}...")
        
        # Step 1: Analyze query type
        query_analysis = self._analyze_query(query)
        
        # Step 2: Fuse and rank context
        fused_context = self._fuse_context(
            query, 
            context_documents, 
            max_context_length or self.max_length
        )
        
        # Step 3: Select appropriate template
        template_type = response_type if response_type in self.response_templates else 'default'
        if query_analysis['type'] in self.response_templates:
            template_type = query_analysis['type']
        
        # Step 4: Generate answer
        answer = self._synthesize_answer(query, fused_context, query_analysis)
        
        # Step 5: Apply template
        response = self._apply_template(
            template_type, 
            query, 
            fused_context['text'], 
            answer
        )
        
        # Step 6: Post-process response
        final_response = self._post_process_response(response)
        
        return {
            'response': final_response,
            'query_analysis': query_analysis,
            'context_used': fused_context,
            'template_type': template_type,
            'confidence': self._calculate_confidence(query, context_documents, answer),
            'metadata': {
                'num_context_docs': len(context_documents),
                'context_length': len(fused_context['text']),
                'response_length': len(final_response)
            }
        }
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine type and characteristics."""
        query_lower = query.lower()
        
        # Query type detection
        query_type = 'default'
        if any(word in query_lower for word in ['what is', 'define', 'explain']):
            query_type = 'explanatory'
        elif any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs']):
            query_type = 'comparative'
        elif any(word in query_lower for word in ['analyze', 'analysis', 'why', 'how']):
            query_type = 'analytical'
        elif any(word in query_lower for word in ['fact', 'when', 'where', 'who']):
            query_type = 'factual'
        
        # Extract key entities/concepts
        entities = self._extract_entities(query)
        
        # Determine complexity
        complexity = 'simple' if len(query.split()) < 10 else 'complex'
        
        return {
            'type': query_type,
            'entities': entities,
            'complexity': complexity,
            'length': len(query),
            'word_count': len(query.split())
        }
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract key entities from text using simple heuristics."""
        # Remove common words and extract potential entities
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'how', 'when', 'where', 'why', 'who'}
        
        words = re.findall(r'\b[A-Za-z]+\b', text.lower())
        entities = [word for word in words if word not in common_words and len(word) > 2]
        
        # Return unique entities, limited to top 10
        return list(set(entities))[:10]
    
    def _fuse_context(self, 
                     query: str, 
                     context_documents: List[Dict[str, Any]], 
                     max_length: int) -> Dict[str, Any]:
        """
        Fuse context documents using advanced techniques.
        
        Args:
            query: User query
            context_documents: List of context documents
            max_length: Maximum context length
            
        Returns:
            Fused context with metadata
        """
        if not context_documents:
            return {'text': '', 'sources': [], 'fusion_method': 'none'}
        
        # Limit to top chunks
        top_docs = context_documents[:self.max_context_chunks]
        
        if self.context_fusion_method == 'weighted_combination':
            return self._weighted_context_fusion(query, top_docs, max_length)
        elif self.context_fusion_method == 'hierarchical':
            return self._hierarchical_context_fusion(query, top_docs, max_length)
        else:
            return self._simple_context_fusion(top_docs, max_length)
    
    def _weighted_context_fusion(self, 
                                query: str, 
                                documents: List[Dict[str, Any]], 
                                max_length: int) -> Dict[str, Any]:
        """Fuse context using weighted combination based on relevance."""
        # Calculate query-document similarities for weighting
        doc_texts = [doc['document'] for doc in documents]
        
        # Use TF-IDF for similarity calculation
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            all_texts = [query] + doc_texts
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            query_vector = tfidf_matrix[0:1]
            doc_vectors = tfidf_matrix[1:]
            
            similarities = cosine_similarity(query_vector, doc_vectors).flatten()
        except:
            # Fallback to original scores
            similarities = np.array([doc.get('score', 0.5) for doc in documents])
        
        # Normalize similarities to use as weights
        weights = similarities / similarities.sum() if similarities.sum() > 0 else np.ones(len(similarities)) / len(similarities)
        
        # Create weighted context
        fused_text = ""
        sources = []
        current_length = 0
        
        # Sort by weight (descending)
        sorted_indices = np.argsort(weights)[::-1]
        
        for i in sorted_indices:
            doc = documents[i]
            weight = weights[i]
            text = doc['document']
            
            # Add document with weight indicator
            doc_text = f"[Relevance: {weight:.2f}] {text}"
            
            if current_length + len(doc_text) <= max_length:
                fused_text += doc_text + "\n\n"
                current_length += len(doc_text) + 2
                sources.append({
                    'index': i,
                    'weight': weight,
                    'score': doc.get('score', 0),
                    'metadata': doc.get('metadata', {})
                })
            else:
                # Truncate if needed
                remaining_space = max_length - current_length
                if remaining_space > 100:  # Only add if meaningful space left
                    truncated_text = doc_text[:remaining_space-3] + "..."
                    fused_text += truncated_text
                    sources.append({
                        'index': i,
                        'weight': weight,
                        'score': doc.get('score', 0),
                        'metadata': doc.get('metadata', {}),
                        'truncated': True
                    })
                break
        
        return {
            'text': fused_text.strip(),
            'sources': sources,
            'fusion_method': 'weighted_combination',
            'total_weight': weights.sum()
        }
    
    def _simple_context_fusion(self, 
                              documents: List[Dict[str, Any]], 
                              max_length: int) -> Dict[str, Any]:
        """Simple context fusion by concatenation."""
        fused_text = ""
        sources = []
        current_length = 0
        
        for i, doc in enumerate(documents):
            text = doc['document']
            
            if current_length + len(text) <= max_length:
                fused_text += text + "\n\n"
                current_length += len(text) + 2
                sources.append({
                    'index': i,
                    'score': doc.get('score', 0),
                    'metadata': doc.get('metadata', {})
                })
            else:
                # Truncate if needed
                remaining_space = max_length - current_length
                if remaining_space > 100:
                    truncated_text = text[:remaining_space-3] + "..."
                    fused_text += truncated_text
                    sources.append({
                        'index': i,
                        'score': doc.get('score', 0),
                        'metadata': doc.get('metadata', {}),
                        'truncated': True
                    })
                break
        
        return {
            'text': fused_text.strip(),
            'sources': sources,
            'fusion_method': 'simple'
        }

    def _hierarchical_context_fusion(self,
                                   query: str,
                                   documents: List[Dict[str, Any]],
                                   max_length: int) -> Dict[str, Any]:
        """Hierarchical context fusion organizing by relevance tiers."""
        # Group documents by score ranges
        high_relevance = [doc for doc in documents if doc.get('score', 0) > 0.7]
        medium_relevance = [doc for doc in documents if 0.3 <= doc.get('score', 0) <= 0.7]
        low_relevance = [doc for doc in documents if doc.get('score', 0) < 0.3]

        fused_text = ""
        sources = []
        current_length = 0

        # Add high relevance first
        if high_relevance:
            fused_text += "High Relevance Information:\n"
            current_length += len("High Relevance Information:\n")

            for doc in high_relevance:
                text = f"• {doc['document']}\n"
                if current_length + len(text) <= max_length:
                    fused_text += text
                    current_length += len(text)
                    sources.append({'tier': 'high', 'document': doc})

        # Add medium relevance if space allows
        if medium_relevance and current_length < max_length * 0.8:
            fused_text += "\nMedium Relevance Information:\n"
            current_length += len("\nMedium Relevance Information:\n")

            for doc in medium_relevance:
                text = f"• {doc['document']}\n"
                if current_length + len(text) <= max_length:
                    fused_text += text
                    current_length += len(text)
                    sources.append({'tier': 'medium', 'document': doc})

        return {
            'text': fused_text.strip(),
            'sources': sources,
            'fusion_method': 'hierarchical'
        }

    def _synthesize_answer(self,
                          query: str,
                          fused_context: Dict[str, Any],
                          query_analysis: Dict[str, Any]) -> str:
        """
        Synthesize an answer from the query and context using rule-based approach.
        """
        context_text = fused_context['text']

        if not context_text.strip():
            return "I don't have enough information to answer this query."

        # Extract key information based on query type
        if query_analysis['type'] == 'factual':
            return self._extract_factual_answer(query, context_text, query_analysis)
        elif query_analysis['type'] == 'explanatory':
            return self._generate_explanation(query, context_text, query_analysis)
        elif query_analysis['type'] == 'comparative':
            return self._generate_comparison(query, context_text, query_analysis)
        elif query_analysis['type'] == 'analytical':
            return self._generate_analysis(query, context_text, query_analysis)
        else:
            return self._generate_default_answer(query, context_text, query_analysis)

    def _extract_factual_answer(self, query: str, context: str, analysis: Dict[str, Any]) -> str:
        """Extract factual information from context."""
        # Look for direct answers to factual questions
        query_entities = analysis.get('entities', [])

        # Find sentences containing query entities
        sentences = re.split(r'[.!?]+', context)
        relevant_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and any(entity in sentence.lower() for entity in query_entities):
                relevant_sentences.append(sentence)

        if relevant_sentences:
            # Return the most relevant sentences
            return '. '.join(relevant_sentences[:3]) + '.'
        else:
            # Fallback to first few sentences
            return '. '.join(sentences[:2]) + '.' if sentences else "No specific factual information found."

    def _generate_explanation(self, query: str, context: str, analysis: Dict[str, Any]) -> str:
        """Generate an explanatory answer."""
        # Structure explanation with definition and details
        sentences = [s.strip() for s in re.split(r'[.!?]+', context) if s.strip()]

        if not sentences:
            return "No explanatory information available."

        # Try to find definition-like sentences
        definition_sentences = [s for s in sentences if any(word in s.lower() for word in ['is', 'are', 'means', 'refers to', 'defined as'])]

        explanation = ""
        if definition_sentences:
            explanation += definition_sentences[0] + ". "

        # Add supporting details
        remaining_sentences = [s for s in sentences if s not in definition_sentences]
        if remaining_sentences:
            explanation += ' '.join(remaining_sentences[:2]) + "."

        return explanation if explanation else sentences[0] + "."

    def _generate_comparison(self, query: str, context: str, analysis: Dict[str, Any]) -> str:
        """Generate a comparative answer."""
        sentences = [s.strip() for s in re.split(r'[.!?]+', context) if s.strip()]

        if len(sentences) < 2:
            return "Insufficient information for comparison."

        # Look for comparative language
        comparative_sentences = [s for s in sentences if any(word in s.lower() for word in ['different', 'similar', 'compared', 'versus', 'while', 'whereas', 'however'])]

        if comparative_sentences:
            return ' '.join(comparative_sentences[:2]) + "."
        else:
            # Create basic comparison from available information
            return f"Based on the available information: {sentences[0]}. Additionally, {sentences[1]}." if len(sentences) >= 2 else sentences[0] + "."

    def _generate_analysis(self, query: str, context: str, analysis: Dict[str, Any]) -> str:
        """Generate an analytical answer."""
        sentences = [s.strip() for s in re.split(r'[.!?]+', context) if s.strip()]

        if not sentences:
            return "No information available for analysis."

        # Structure analytical response
        analysis_text = f"Analyzing the available information: {sentences[0]}."

        if len(sentences) > 1:
            analysis_text += f" Furthermore, {sentences[1]}."

        if len(sentences) > 2:
            analysis_text += f" This suggests that {sentences[2]}."

        return analysis_text

    def _generate_default_answer(self, query: str, context: str, analysis: Dict[str, Any]) -> str:
        """Generate a default answer."""
        sentences = [s.strip() for s in re.split(r'[.!?]+', context) if s.strip()]

        if not sentences:
            return "No relevant information found."

        # Return most relevant sentences
        return '. '.join(sentences[:3]) + '.' if len(sentences) >= 3 else '. '.join(sentences) + '.'

    def _apply_template(self, template_type: str, query: str, context: str, answer: str) -> str:
        """Apply response template."""
        template = self.response_templates.get(template_type, self.response_templates['default'])

        return template.format(
            query=query,
            context=context,
            answer=answer
        )

    def _post_process_response(self, response: str) -> str:
        """Post-process the response for better formatting."""
        # Clean up extra whitespace
        response = re.sub(r'\n\s*\n', '\n\n', response)
        response = re.sub(r' +', ' ', response)

        # Ensure proper sentence endings
        response = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', response)

        # Limit length if needed
        if len(response) > self.max_length:
            response = response[:self.max_length-3] + "..."

        return response.strip()

    def _calculate_confidence(self,
                            query: str,
                            context_documents: List[Dict[str, Any]],
                            answer: str) -> float:
        """Calculate confidence score for the response."""
        if not context_documents:
            return 0.0

        # Base confidence on average retrieval scores
        avg_score = np.mean([doc.get('score', 0) for doc in context_documents])

        # Adjust based on number of sources
        source_factor = min(len(context_documents) / 3.0, 1.0)  # Normalize to max of 3 sources

        # Adjust based on answer length (longer answers might be more comprehensive)
        length_factor = min(len(answer) / 200.0, 1.0)  # Normalize to 200 chars

        # Combine factors
        confidence = (0.6 * avg_score + 0.3 * source_factor + 0.1 * length_factor)

        return min(confidence, 1.0)
