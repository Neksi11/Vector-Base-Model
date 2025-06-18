"""
Advanced RAG Agent Usage Examples

This script demonstrates the advanced features of the RAG agent including:
- Hybrid search capabilities
- Advanced response generation
- Query processing and expansion
- Comprehensive evaluation metrics
- Performance monitoring
"""

import time
import json
from rag_agent.agent import RAGAgent
from rag_agent.vectorstore import AdvancedVectorStore
from rag_agent.response_generator import ResponseGenerator
from rag_agent.query_processor import QueryProcessor
from rag_agent.evaluator import RAGEvaluator


def create_sample_dataset():
    """Create a comprehensive sample dataset for demonstration."""
    documents = [
        # Machine Learning Documents
        "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make predictions or decisions.",
        
        "Supervised learning is a type of machine learning where algorithms learn from labeled training data. Common supervised learning tasks include classification (predicting categories) and regression (predicting continuous values).",
        
        "Unsupervised learning discovers hidden patterns in data without labeled examples. Clustering, dimensionality reduction, and association rule learning are common unsupervised learning techniques.",
        
        "Deep learning uses artificial neural networks with multiple layers to model and understand complex patterns in data. It has revolutionized fields like computer vision, natural language processing, and speech recognition.",
        
        # Data Science Documents
        "Data science combines statistics, programming, and domain expertise to extract insights from data. It involves data collection, cleaning, analysis, visualization, and interpretation to solve real-world problems.",
        
        "Feature engineering is the process of selecting, modifying, or creating new features from raw data to improve machine learning model performance. Good features can significantly impact model accuracy.",
        
        "Cross-validation is a statistical method used to estimate the performance of machine learning models. It involves partitioning data into subsets, training on some subsets, and testing on others.",
        
        # Algorithms and Techniques
        "Random Forest is an ensemble learning method that combines multiple decision trees to improve prediction accuracy and reduce overfitting. It uses bootstrap aggregating (bagging) and random feature selection.",
        
        "Support Vector Machines (SVM) are powerful supervised learning algorithms used for classification and regression. They find optimal decision boundaries by maximizing the margin between different classes.",
        
        "K-means clustering is an unsupervised learning algorithm that partitions data into k clusters. It iteratively assigns data points to the nearest cluster centroid and updates centroids based on cluster members.",
        
        # Evaluation and Metrics
        "Model evaluation metrics help assess the performance of machine learning algorithms. Common metrics include accuracy, precision, recall, F1-score for classification, and MSE, RMSE for regression.",
        
        "Overfitting occurs when a model learns the training data too well, including noise and irrelevant patterns. This leads to poor generalization on new, unseen data.",
        
        "Bias-variance tradeoff is a fundamental concept in machine learning. High bias leads to underfitting, while high variance leads to overfitting. The goal is to find the optimal balance.",
        
        # Applications
        "Natural Language Processing (NLP) applies machine learning to understand and generate human language. Applications include sentiment analysis, machine translation, chatbots, and text summarization.",
        
        "Computer vision uses machine learning to interpret and analyze visual information from images and videos. Applications include object detection, facial recognition, and medical image analysis.",
        
        "Recommendation systems use machine learning to suggest relevant items to users based on their preferences and behavior. They are widely used in e-commerce, streaming services, and social media."
    ]
    
    labels = [
        "ml_basics", "supervised_learning", "unsupervised_learning", "deep_learning",
        "data_science", "feature_engineering", "model_validation", 
        "ensemble_methods", "svm", "clustering",
        "evaluation", "overfitting", "bias_variance",
        "nlp", "computer_vision", "recommendation_systems"
    ]
    
    metadata = [
        {"category": "fundamentals", "difficulty": "beginner", "topic": "machine_learning"},
        {"category": "algorithms", "difficulty": "intermediate", "topic": "supervised_learning"},
        {"category": "algorithms", "difficulty": "intermediate", "topic": "unsupervised_learning"},
        {"category": "algorithms", "difficulty": "advanced", "topic": "neural_networks"},
        {"category": "fundamentals", "difficulty": "beginner", "topic": "data_science"},
        {"category": "preprocessing", "difficulty": "intermediate", "topic": "feature_engineering"},
        {"category": "evaluation", "difficulty": "intermediate", "topic": "validation"},
        {"category": "algorithms", "difficulty": "intermediate", "topic": "ensemble"},
        {"category": "algorithms", "difficulty": "intermediate", "topic": "classification"},
        {"category": "algorithms", "difficulty": "beginner", "topic": "clustering"},
        {"category": "evaluation", "difficulty": "intermediate", "topic": "metrics"},
        {"category": "concepts", "difficulty": "intermediate", "topic": "model_issues"},
        {"category": "concepts", "difficulty": "advanced", "topic": "theory"},
        {"category": "applications", "difficulty": "intermediate", "topic": "nlp"},
        {"category": "applications", "difficulty": "intermediate", "topic": "vision"},
        {"category": "applications", "difficulty": "intermediate", "topic": "recommendation"}
    ]
    
    return documents, labels, metadata


def demonstrate_basic_usage():
    """Demonstrate basic RAG agent usage."""
    print("=== Basic RAG Agent Usage ===")
    
    # Create sample data
    documents, labels, metadata = create_sample_dataset()
    
    # Initialize agent
    agent = RAGAgent(
        max_context_length=2000,
        top_k_retrieval=5,
        enable_query_expansion=True,
        enable_reranking=True
    )
    
    # Ingest documents
    print(f"Ingesting {len(documents)} documents...")
    agent.ingest_documents(documents, labels, metadata)
    
    # Test queries
    queries = [
        "What is machine learning?",
        "How does supervised learning work?",
        "Explain the difference between bias and variance",
        "What are some applications of machine learning?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        start_time = time.time()
        
        response = agent.generate_response(
            query,
            return_sources=True,
            return_confidence=True
        )
        
        print(f"Response: {response['response']}")
        print(f"Confidence: {response['confidence']:.3f}")
        print(f"Sources used: {len(response['sources'])}")
        print(f"Response time: {time.time() - start_time:.3f}s")


def demonstrate_advanced_features():
    """Demonstrate advanced RAG features."""
    print("\n=== Advanced RAG Features ===")
    
    # Create advanced vector store
    vectorstore = AdvancedVectorStore(
        chunk_size=300,
        chunk_overlap=50,
        use_hybrid_search=True,
        similarity_metrics=['cosine', 'euclidean']
    )
    
    # Create sample data
    documents, _, metadata = create_sample_dataset()
    
    # Add documents to vector store
    vectorstore.add_documents(documents, metadata)
    
    # Test different search types
    query = "machine learning algorithms for classification"
    
    search_types = ['hybrid', 'dense', 'sparse', 'ensemble']
    
    for search_type in search_types:
        print(f"\n--- {search_type.upper()} Search ---")
        results = vectorstore.similarity_search(
            query, 
            top_k=3, 
            search_type=search_type
        )
        
        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result['score']:.3f}")
            print(f"   Text: {result['document'][:100]}...")
            print(f"   Metadata: {result['metadata']}")


def demonstrate_query_processing():
    """Demonstrate advanced query processing."""
    print("\n=== Query Processing Features ===")
    
    processor = QueryProcessor(
        enable_synonym_expansion=True,
        max_expanded_terms=3
    )
    
    test_queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "Compare supervised vs unsupervised learning",
        "Explain the bias-variance tradeoff in ML models"
    ]
    
    for query in test_queries:
        print(f"\nOriginal Query: {query}")
        
        processed = processor.process_query(query, expand=True, analyze_intent=True)
        
        print(f"Cleaned: {processed['cleaned_query']}")
        print(f"Expanded: {processed['expanded_query']}")
        print(f"Intent: {processed['intent']['primary_intent']}")
        print(f"Keywords: {processed['keywords']}")
        print(f"Entities: {processed['entities']}")
        print(f"Complexity: {processed['complexity_score']:.3f}")


def demonstrate_response_generation():
    """Demonstrate advanced response generation."""
    print("\n=== Response Generation Features ===")
    
    generator = ResponseGenerator(
        max_length=1000,
        temperature=0.7
    )
    
    # Sample context documents
    context_docs = [
        {
            'document': "Machine learning is a subset of AI that enables computers to learn from data without explicit programming.",
            'score': 0.95,
            'metadata': {'category': 'fundamentals'}
        },
        {
            'document': "Supervised learning uses labeled data to train models for prediction tasks like classification and regression.",
            'score': 0.87,
            'metadata': {'category': 'algorithms'}
        },
        {
            'document': "Deep learning uses neural networks with multiple layers to model complex patterns in data.",
            'score': 0.82,
            'metadata': {'category': 'advanced'}
        }
    ]
    
    query = "What is machine learning and how does it work?"
    
    response_types = ['factual', 'explanatory', 'analytical']
    
    for response_type in response_types:
        print(f"\n--- {response_type.upper()} Response ---")
        
        result = generator.generate_response(
            query=query,
            context_documents=context_docs,
            response_type=response_type
        )
        
        print(f"Response: {result['response']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Template: {result['template_type']}")


def demonstrate_evaluation():
    """Demonstrate comprehensive evaluation."""
    print("\n=== Evaluation and Metrics ===")
    
    evaluator = RAGEvaluator()
    
    # Sample evaluation data
    query = "What is machine learning?"
    response = "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make predictions."
    
    retrieved_docs = [
        {
            'document': "Machine learning is a subset of AI that enables computers to learn from data.",
            'score': 0.95
        },
        {
            'document': "AI algorithms can analyze data to identify patterns and make predictions.",
            'score': 0.87
        }
    ]
    
    ground_truth = "Machine learning is a branch of artificial intelligence that focuses on algorithms that can learn from and make predictions on data."
    
    # Perform evaluation
    start_time = time.time()
    evaluation = evaluator.evaluate_response(
        query=query,
        response=response,
        retrieved_docs=retrieved_docs,
        ground_truth=ground_truth,
        start_time=start_time
    )
    
    print("Evaluation Results:")
    print(f"Overall Score: {evaluation['overall_score']:.3f}")
    print(f"Retrieval Quality: {evaluation['retrieval_metrics']['avg_score']:.3f}")
    print(f"Response Relevance: {evaluation['relevance_metrics']['query_relevance']:.3f}")
    print(f"Response Quality: {evaluation['quality_metrics']['coherence_score']:.3f}")
    print(f"Ground Truth Similarity: {evaluation['ground_truth_metrics']['semantic_similarity']:.3f}")
    
    # Get performance summary
    summary = evaluator.get_performance_summary()
    print(f"\nPerformance Summary:")
    print(f"Total Queries: {summary['current_metrics']['total_queries']}")
    print(f"Average Confidence: {summary['current_metrics']['avg_confidence']:.3f}")


def demonstrate_end_to_end_workflow():
    """Demonstrate complete end-to-end workflow."""
    print("\n=== End-to-End Workflow ===")
    
    # Initialize advanced agent
    agent = RAGAgent(
        max_context_length=1500,
        top_k_retrieval=8,
        response_temperature=0.6,
        enable_query_expansion=True,
        enable_reranking=True
    )
    
    # Create and ingest data
    documents, labels, metadata = create_sample_dataset()
    agent.ingest_documents(documents, labels, metadata)
    
    # Complex query requiring advanced processing
    complex_query = "Compare supervised and unsupervised learning approaches, explaining their key differences, use cases, and provide examples of algorithms for each category"
    
    print(f"Complex Query: {complex_query}")
    
    # Generate comprehensive response
    start_time = time.time()
    result = agent.generate_response(
        query=complex_query,
        context_filter={'category': 'algorithms'},  # Filter by algorithm-related documents
        return_sources=True,
        return_confidence=True
    )
    
    print(f"\nGenerated Response:")
    print(result['response'])
    
    print(f"\nQuery Analysis:")
    print(f"- Intent: {result['query_analysis']['intent']['primary_intent']}")
    print(f"- Complexity: {result['query_analysis']['complexity_score']:.3f}")
    print(f"- Keywords: {result['query_analysis']['keywords']}")
    
    print(f"\nEvaluation Metrics:")
    print(f"- Overall Score: {result['evaluation']['overall_score']:.3f}")
    print(f"- Retrieval Quality: {result['evaluation']['retrieval_metrics']['avg_score']:.3f}")
    print(f"- Response Relevance: {result['evaluation']['relevance_metrics']['query_relevance']:.3f}")
    
    print(f"\nSources Used ({len(result['sources'])}):")
    for i, source in enumerate(result['sources'][:3], 1):
        print(f"{i}. Score: {source['score']:.3f}, Category: {source['metadata'].get('category', 'N/A')}")
        print(f"   Text: {source['document'][:100]}...")
    
    print(f"\nResponse Time: {time.time() - start_time:.3f}s")
    print(f"Confidence: {result['confidence']:.3f}")
    
    # Get agent statistics
    stats = agent.get_agent_stats()
    print(f"\nAgent Statistics:")
    print(f"- Documents: {stats['num_documents']}")
    print(f"- Chunks: {stats['vectorstore_stats']['num_chunks']}")
    print(f"- Average chunks per document: {stats['vectorstore_stats']['avg_chunks_per_doc']:.1f}")


def main():
    """Run all demonstration examples."""
    print("Advanced RAG Agent Demonstration")
    print("=" * 50)
    
    try:
        demonstrate_basic_usage()
        demonstrate_advanced_features()
        demonstrate_query_processing()
        demonstrate_response_generation()
        demonstrate_evaluation()
        demonstrate_end_to_end_workflow()
        
        print("\n" + "=" * 50)
        print("All demonstrations completed successfully!")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
