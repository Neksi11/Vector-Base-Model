"""
Advanced RAG Agent with sophisticated response generation capabilities.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from .vectorstore import VectorStore
from .classifier import DocumentClassifier
from .response_generator import ResponseGenerator
from .query_processor import QueryProcessor
from .evaluator import RAGEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGAgent:
    """
    Advanced Retrieval Augmented Generation Agent with multi-step reasoning,
    context ranking, and sophisticated response generation.
    """
    
    def __init__(self, 
                 max_context_length: int = 4000,
                 top_k_retrieval: int = 10,
                 response_temperature: float = 0.7,
                 enable_query_expansion: bool = True,
                 enable_reranking: bool = True):
        """
        Initialize the advanced RAG agent.
        
        Args:
            max_context_length: Maximum context length for response generation
            top_k_retrieval: Number of documents to retrieve initially
            response_temperature: Temperature for response generation
            enable_query_expansion: Whether to expand queries for better retrieval
            enable_reranking: Whether to rerank retrieved documents
        """
        self.max_context_length = max_context_length
        self.top_k_retrieval = top_k_retrieval
        self.response_temperature = response_temperature
        self.enable_query_expansion = enable_query_expansion
        self.enable_reranking = enable_reranking
        
        # Initialize components
        self.vectorstore = VectorStore()
        self.classifier = DocumentClassifier()
        self.response_generator = ResponseGenerator(
            max_length=max_context_length,
            temperature=response_temperature
        )
        self.query_processor = QueryProcessor()
        self.evaluator = RAGEvaluator()
        
        # State tracking
        self.is_trained = False
        self.document_metadata = []
        
        logger.info("Advanced RAG Agent initialized")
    
    def ingest_documents(self, 
                        documents: List[str], 
                        labels: Optional[List[str]] = None,
                        metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Ingest documents with optional labels and metadata.
        
        Args:
            documents: List of document texts
            labels: Optional labels for classification training
            metadata: Optional metadata for each document
        """
        logger.info(f"Ingesting {len(documents)} documents")
        
        # Store metadata
        if metadata:
            self.document_metadata.extend(metadata)
        else:
            self.document_metadata.extend([{} for _ in documents])
        
        # Add documents to vector store
        self.vectorstore.add_documents(documents)
        
        # Train classifier if labels provided
        if labels:
            self.classifier.train(documents, labels)
            logger.info("Classifier trained on document labels")
        
        self.is_trained = True
        logger.info("Document ingestion completed")
    
    def generate_response(self, 
                         query: str,
                         context_filter: Optional[Dict[str, Any]] = None,
                         return_sources: bool = True,
                         return_confidence: bool = True) -> Dict[str, Any]:
        """
        Generate a comprehensive response to a query with advanced processing.
        
        Args:
            query: User query
            context_filter: Optional filters for document retrieval
            return_sources: Whether to return source documents
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary containing response, sources, confidence, and metadata
        """
        if not self.is_trained:
            raise ValueError("Agent must be trained before generating responses")
        
        logger.info(f"Processing query: {query}")
        
        # Step 1: Process and analyze query
        processed_query = self.query_processor.process_query(
            query, 
            expand=self.enable_query_expansion
        )
        
        # Step 2: Retrieve relevant documents
        retrieved_docs = self._retrieve_documents(
            processed_query, 
            context_filter
        )
        
        # Step 3: Rerank documents if enabled
        if self.enable_reranking and len(retrieved_docs) > 1:
            retrieved_docs = self._rerank_documents(query, retrieved_docs)
        
        # Step 4: Generate response
        response_data = self.response_generator.generate_response(
            query=query,
            context_documents=retrieved_docs,
            max_context_length=self.max_context_length
        )
        
        # Step 5: Evaluate response quality
        evaluation_metrics = self.evaluator.evaluate_response(
            query=query,
            response=response_data['response'],
            retrieved_docs=retrieved_docs
        )
        
        # Compile final response
        result = {
            'response': response_data['response'],
            'query_analysis': processed_query,
            'evaluation': evaluation_metrics
        }
        
        if return_sources:
            result['sources'] = [
                {
                    'document': doc['document'],
                    'score': doc['score'],
                    'metadata': self.document_metadata[doc.get('index', 0)]
                }
                for doc in retrieved_docs
            ]
        
        if return_confidence:
            result['confidence'] = response_data.get('confidence', 0.0)
        
        logger.info("Response generation completed")
        return result
    
    def _retrieve_documents(self, 
                           processed_query: Dict[str, Any],
                           context_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant documents using advanced search."""
        # Use expanded query if available
        search_query = processed_query.get('expanded_query', processed_query['original_query'])
        
        # Perform similarity search
        results = self.vectorstore.similarity_search(
            search_query, 
            top_k=self.top_k_retrieval
        )
        
        # Apply context filters if provided
        if context_filter:
            results = self._apply_context_filter(results, context_filter)
        
        return results
    
    def _rerank_documents(self, 
                         query: str, 
                         documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank documents using advanced scoring."""
        # Use classifier predictions if available
        if hasattr(self.classifier, 'pipeline') and self.classifier.pipeline:
            doc_texts = [doc['document'] for doc in documents]
            predictions = self.classifier.predict_proba(doc_texts)
            
            # Combine similarity scores with classification confidence
            for i, doc in enumerate(documents):
                if i < len(predictions):
                    class_confidence = max(predictions[i])
                    doc['combined_score'] = 0.7 * doc['score'] + 0.3 * class_confidence
                else:
                    doc['combined_score'] = doc['score']
            
            # Sort by combined score
            documents.sort(key=lambda x: x.get('combined_score', x['score']), reverse=True)
        
        return documents
    
    def _apply_context_filter(self, 
                             documents: List[Dict[str, Any]], 
                             context_filter: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply context-based filtering to retrieved documents."""
        filtered_docs = []
        
        for doc in documents:
            doc_index = doc.get('index', 0)
            if doc_index < len(self.document_metadata):
                metadata = self.document_metadata[doc_index]
                
                # Check if document matches filter criteria
                matches_filter = True
                for key, value in context_filter.items():
                    if key not in metadata or metadata[key] != value:
                        matches_filter = False
                        break
                
                if matches_filter:
                    filtered_docs.append(doc)
        
        return filtered_docs
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the agent."""
        return {
            'is_trained': self.is_trained,
            'num_documents': len(self.vectorstore.documents),
            'num_metadata_entries': len(self.document_metadata),
            'vectorstore_stats': self.vectorstore.get_stats(),
            'classifier_trained': hasattr(self.classifier, 'pipeline') and self.classifier.pipeline is not None,
            'configuration': {
                'max_context_length': self.max_context_length,
                'top_k_retrieval': self.top_k_retrieval,
                'response_temperature': self.response_temperature,
                'enable_query_expansion': self.enable_query_expansion,
                'enable_reranking': self.enable_reranking
            }
        }
