"""
Comprehensive test suite for the Advanced RAG Agent system.
"""

import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock

# Import the RAG components
from rag_agent.agent import RAGAgent
from rag_agent.vectorstore import VectorStore, AdvancedVectorStore
from rag_agent.classifier import DocumentClassifier
from rag_agent.response_generator import ResponseGenerator
from rag_agent.query_processor import QueryProcessor
from rag_agent.evaluator import RAGEvaluator


class TestVectorStore(unittest.TestCase):
    """Test cases for VectorStore functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.vectorstore = VectorStore()
        self.sample_documents = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Supervised learning requires labeled training data.",
            "Unsupervised learning finds patterns in unlabeled data."
        ]
    
    def test_add_documents(self):
        """Test adding documents to vector store."""
        self.vectorstore.add_documents(self.sample_documents)
        
        self.assertEqual(len(self.vectorstore.documents), 4)
        self.assertIsNotNone(self.vectorstore.document_vectors)
        self.assertTrue(self.vectorstore.is_fitted)
    
    def test_similarity_search(self):
        """Test similarity search functionality."""
        self.vectorstore.add_documents(self.sample_documents)
        
        results = self.vectorstore.similarity_search("neural networks", top_k=2)
        
        self.assertEqual(len(results), 2)
        self.assertIn('document', results[0])
        self.assertIn('score', results[0])
        self.assertGreater(results[0]['score'], 0)
    
    def test_empty_search(self):
        """Test search with no documents."""
        with self.assertRaises(ValueError):
            self.vectorstore.similarity_search("test query")


class TestAdvancedVectorStore(unittest.TestCase):
    """Test cases for AdvancedVectorStore functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.vectorstore = AdvancedVectorStore(
            chunk_size=100,
            chunk_overlap=20,
            use_hybrid_search=True
        )
        self.sample_documents = [
            "Machine learning is a subset of artificial intelligence that enables computers to learn from data. It includes supervised learning, unsupervised learning, and reinforcement learning approaches.",
            "Deep learning is a specialized form of machine learning that uses artificial neural networks with multiple layers. These networks can automatically learn hierarchical representations of data."
        ]
    
    def test_semantic_chunking(self):
        """Test semantic chunking functionality."""
        self.vectorstore.add_documents(self.sample_documents)
        
        # Should create multiple chunks from longer documents
        self.assertGreater(len(self.vectorstore.chunks), len(self.sample_documents))
        self.assertEqual(len(self.vectorstore.chunk_metadata), len(self.vectorstore.chunks))
    
    def test_hybrid_search(self):
        """Test hybrid search functionality."""
        self.vectorstore.add_documents(self.sample_documents)
        
        results = self.vectorstore.similarity_search(
            "neural networks", 
            top_k=3, 
            search_type='hybrid'
        )
        
        self.assertLessEqual(len(results), 3)
        for result in results:
            self.assertIn('search_type', result)
            self.assertEqual(result['search_type'], 'hybrid')
    
    def test_different_search_types(self):
        """Test different search types."""
        self.vectorstore.add_documents(self.sample_documents)
        
        search_types = ['dense', 'sparse', 'ensemble']
        
        for search_type in search_types:
            results = self.vectorstore.similarity_search(
                "machine learning", 
                top_k=2, 
                search_type=search_type
            )
            
            self.assertLessEqual(len(results), 2)
            if results:
                self.assertEqual(results[0]['search_type'], search_type)


class TestDocumentClassifier(unittest.TestCase):
    """Test cases for DocumentClassifier."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.classifier = DocumentClassifier()
        self.sample_documents = [
            "Machine learning algorithms for classification",
            "Data preprocessing and feature engineering",
            "Model evaluation and validation techniques",
            "Deep learning and neural networks"
        ]
        self.sample_labels = ["algorithm", "preprocessing", "evaluation", "deep_learning"]
    
    def test_training(self):
        """Test classifier training."""
        self.classifier.train(self.sample_documents, self.sample_labels)
        
        # Check that pipeline is fitted
        self.assertIsNotNone(self.classifier.pipeline)
    
    def test_prediction(self):
        """Test classifier prediction."""
        self.classifier.train(self.sample_documents, self.sample_labels)
        
        test_docs = ["Random forest classification algorithm"]
        predictions = self.classifier.predict(test_docs)
        
        self.assertEqual(len(predictions), 1)
        self.assertIn(predictions[0], self.sample_labels)
    
    def test_prediction_probabilities(self):
        """Test prediction probabilities."""
        self.classifier.train(self.sample_documents, self.sample_labels)
        
        test_docs = ["Neural network architecture"]
        probabilities = self.classifier.predict_proba(test_docs)
        
        self.assertEqual(len(probabilities), 1)
        self.assertEqual(len(probabilities[0]), len(set(self.sample_labels)))


class TestQueryProcessor(unittest.TestCase):
    """Test cases for QueryProcessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = QueryProcessor(
            enable_synonym_expansion=True,
            max_expanded_terms=3
        )
    
    def test_query_cleaning(self):
        """Test query cleaning functionality."""
        dirty_query = "  What is   machine learning???  "
        processed = self.processor.process_query(dirty_query)
        
        self.assertEqual(processed['cleaned_query'], "what is machine learning?")
        self.assertNotEqual(processed['cleaned_query'], dirty_query)
    
    def test_intent_detection(self):
        """Test intent detection."""
        queries_and_intents = [
            ("What is machine learning?", "definition"),
            ("How does neural network work?", "explanation"),
            ("Compare supervised vs unsupervised learning", "comparison"),
            ("When was AI invented?", "factual")
        ]
        
        for query, expected_intent in queries_and_intents:
            processed = self.processor.process_query(query)
            detected_intent = processed['intent']['primary_intent']
            
            # Intent detection should work for clear cases
            if expected_intent != "general":
                self.assertEqual(detected_intent, expected_intent)
    
    def test_keyword_extraction(self):
        """Test keyword extraction."""
        query = "What is machine learning algorithm classification?"
        processed = self.processor.process_query(query)
        
        keywords = processed['keywords']
        self.assertIn('machine', keywords)
        self.assertIn('learning', keywords)
        self.assertIn('algorithm', keywords)
        self.assertNotIn('what', keywords)  # Should remove question words
    
    def test_query_expansion(self):
        """Test query expansion."""
        query = "machine learning algorithm"
        processed = self.processor.process_query(query, expand=True)
        
        expanded_query = processed['expanded_query']
        expansion_terms = processed['expansion_terms']
        
        self.assertGreaterEqual(len(expanded_query), len(query))
        if expansion_terms:
            self.assertGreater(len(expansion_terms), 0)


class TestResponseGenerator(unittest.TestCase):
    """Test cases for ResponseGenerator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = ResponseGenerator(max_length=500)
        self.sample_context = [
            {
                'document': "Machine learning is a subset of AI that enables computers to learn from data.",
                'score': 0.9,
                'metadata': {'category': 'fundamentals'}
            },
            {
                'document': "Supervised learning uses labeled data for training predictive models.",
                'score': 0.8,
                'metadata': {'category': 'algorithms'}
            }
        ]
    
    def test_response_generation(self):
        """Test basic response generation."""
        query = "What is machine learning?"
        
        result = self.generator.generate_response(
            query=query,
            context_documents=self.sample_context
        )
        
        self.assertIn('response', result)
        self.assertIn('confidence', result)
        self.assertGreater(len(result['response']), 0)
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)
    
    def test_different_response_types(self):
        """Test different response types."""
        query = "Explain machine learning"
        response_types = ['factual', 'explanatory', 'analytical']
        
        for response_type in response_types:
            result = self.generator.generate_response(
                query=query,
                context_documents=self.sample_context,
                response_type=response_type
            )
            
            self.assertIn('response', result)
            self.assertIn('template_type', result)
    
    def test_empty_context(self):
        """Test response generation with empty context."""
        query = "What is machine learning?"
        
        result = self.generator.generate_response(
            query=query,
            context_documents=[]
        )
        
        self.assertIn('response', result)
        # Should handle empty context gracefully


class TestRAGEvaluator(unittest.TestCase):
    """Test cases for RAGEvaluator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = RAGEvaluator()
        self.sample_query = "What is machine learning?"
        self.sample_response = "Machine learning is a subset of AI that enables computers to learn from data."
        self.sample_docs = [
            {
                'document': "Machine learning is part of artificial intelligence.",
                'score': 0.9
            }
        ]
    
    def test_response_evaluation(self):
        """Test comprehensive response evaluation."""
        evaluation = self.evaluator.evaluate_response(
            query=self.sample_query,
            response=self.sample_response,
            retrieved_docs=self.sample_docs
        )
        
        # Check that all evaluation components are present
        self.assertIn('retrieval_metrics', evaluation)
        self.assertIn('relevance_metrics', evaluation)
        self.assertIn('quality_metrics', evaluation)
        self.assertIn('overall_score', evaluation)
        
        # Check score ranges
        self.assertGreaterEqual(evaluation['overall_score'], 0.0)
        self.assertLessEqual(evaluation['overall_score'], 1.0)
    
    def test_ground_truth_evaluation(self):
        """Test evaluation against ground truth."""
        ground_truth = "Machine learning is a branch of AI focused on algorithms that learn from data."
        
        evaluation = self.evaluator.evaluate_response(
            query=self.sample_query,
            response=self.sample_response,
            retrieved_docs=self.sample_docs,
            ground_truth=ground_truth
        )
        
        self.assertIn('ground_truth_metrics', evaluation)
        self.assertIn('semantic_similarity', evaluation['ground_truth_metrics'])
    
    def test_performance_tracking(self):
        """Test performance metrics tracking."""
        # Perform multiple evaluations
        for i in range(3):
            self.evaluator.evaluate_response(
                query=f"Query {i}",
                response=f"Response {i}",
                retrieved_docs=self.sample_docs
            )
        
        summary = self.evaluator.get_performance_summary()
        
        self.assertEqual(summary['current_metrics']['total_queries'], 3)
        self.assertIn('recent_performance', summary)


class TestRAGAgent(unittest.TestCase):
    """Test cases for the main RAGAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = RAGAgent(
            max_context_length=1000,
            top_k_retrieval=3,
            enable_query_expansion=True
        )
        self.sample_documents = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Supervised learning requires labeled training data."
        ]
        self.sample_labels = ["ml_basics", "deep_learning", "supervised"]
    
    def test_document_ingestion(self):
        """Test document ingestion."""
        self.agent.ingest_documents(self.sample_documents, self.sample_labels)
        
        self.assertTrue(self.agent.is_trained)
        self.assertEqual(len(self.agent.vectorstore.documents), 3)
    
    def test_response_generation(self):
        """Test end-to-end response generation."""
        self.agent.ingest_documents(self.sample_documents, self.sample_labels)
        
        response = self.agent.generate_response(
            "What is machine learning?",
            return_sources=True,
            return_confidence=True
        )
        
        self.assertIn('response', response)
        self.assertIn('sources', response)
        self.assertIn('confidence', response)
        self.assertIn('evaluation', response)
        
        self.assertGreater(len(response['response']), 0)
        self.assertGreaterEqual(response['confidence'], 0.0)
        self.assertLessEqual(response['confidence'], 1.0)
    
    def test_context_filtering(self):
        """Test context filtering functionality."""
        metadata = [
            {"category": "basics", "level": "beginner"},
            {"category": "advanced", "level": "expert"},
            {"category": "basics", "level": "intermediate"}
        ]
        
        self.agent.ingest_documents(self.sample_documents, self.sample_labels, metadata)
        
        # Test filtering by category
        response = self.agent.generate_response(
            "What is machine learning?",
            context_filter={"category": "basics"}
        )
        
        self.assertIn('response', response)
    
    def test_agent_stats(self):
        """Test agent statistics."""
        self.agent.ingest_documents(self.sample_documents, self.sample_labels)
        
        stats = self.agent.get_agent_stats()
        
        self.assertIn('is_trained', stats)
        self.assertIn('num_documents', stats)
        self.assertIn('configuration', stats)
        
        self.assertTrue(stats['is_trained'])
        self.assertEqual(stats['num_documents'], 3)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete RAG system."""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Create agent with all features enabled
        agent = RAGAgent(
            max_context_length=1500,
            top_k_retrieval=5,
            enable_query_expansion=True,
            enable_reranking=True
        )
        
        # Sample dataset
        documents = [
            "Machine learning is a method of data analysis that automates analytical model building.",
            "Artificial intelligence is intelligence demonstrated by machines, in contrast to natural intelligence.",
            "Deep learning is part of a broader family of machine learning methods based on neural networks.",
            "Supervised learning is the machine learning task of learning a function that maps input to output.",
            "Unsupervised learning is a type of machine learning that looks for patterns in data."
        ]
        
        labels = ["ml", "ai", "dl", "supervised", "unsupervised"]
        metadata = [
            {"topic": "machine_learning", "complexity": "medium"},
            {"topic": "artificial_intelligence", "complexity": "low"},
            {"topic": "deep_learning", "complexity": "high"},
            {"topic": "supervised_learning", "complexity": "medium"},
            {"topic": "unsupervised_learning", "complexity": "medium"}
        ]
        
        # Ingest documents
        agent.ingest_documents(documents, labels, metadata)
        
        # Test complex query
        complex_query = "Compare machine learning and artificial intelligence, explaining their relationship and key differences"
        
        response = agent.generate_response(
            complex_query,
            return_sources=True,
            return_confidence=True
        )
        
        # Verify response structure
        self.assertIn('response', response)
        self.assertIn('query_analysis', response)
        self.assertIn('evaluation', response)
        self.assertIn('sources', response)
        self.assertIn('confidence', response)
        
        # Verify response quality
        self.assertGreater(len(response['response']), 50)  # Substantial response
        self.assertGreater(response['confidence'], 0.0)
        self.assertGreater(len(response['sources']), 0)
        
        # Verify evaluation metrics
        evaluation = response['evaluation']
        self.assertIn('overall_score', evaluation)
        self.assertGreaterEqual(evaluation['overall_score'], 0.0)
        self.assertLessEqual(evaluation['overall_score'], 1.0)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestVectorStore,
        TestAdvancedVectorStore,
        TestDocumentClassifier,
        TestQueryProcessor,
        TestResponseGenerator,
        TestRAGEvaluator,
        TestRAGAgent,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
