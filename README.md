# Advanced Vector-Based RAG Agent

A sophisticated Retrieval Augmented Generation (RAG) system with advanced response generation capabilities, built using scikit-learn and modern machine learning techniques.

## 🚀 Features

### Core Capabilities
- **Advanced Vector Store**: Semantic chunking, hybrid search (dense + sparse), multiple similarity metrics
- **Sophisticated Response Generation**: Template-based generation, context fusion, answer synthesis
- **Intelligent Query Processing**: Intent detection, query expansion, semantic analysis
- **Comprehensive Evaluation**: Retrieval quality, response relevance, performance metrics
- **Multi-Modal Search**: Hybrid, dense, sparse, and ensemble search strategies

### Advanced Features
- **Semantic Chunking**: Intelligent text segmentation with configurable overlap
- **Query Expansion**: Automatic synonym expansion and related term addition
- **Context Reranking**: Advanced document reranking using classification confidence
- **Response Templates**: Multiple response types (factual, explanatory, analytical, comparative)
- **Performance Monitoring**: Real-time evaluation and performance tracking
- **Metadata Filtering**: Context-aware document filtering

## 📦 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd vector-base-rag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## 🎯 Quick Start

### Basic Usage

```python
from rag_agent.agent import RAGAgent

# Create sample documents
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Supervised learning requires labeled training data."
]

labels = ["ml_basics", "deep_learning", "supervised"]

# Initialize the RAG agent
agent = RAGAgent(
    max_context_length=2000,
    top_k_retrieval=5,
    enable_query_expansion=True,
    enable_reranking=True
)

# Ingest documents
agent.ingest_documents(documents, labels)

# Generate response
response = agent.generate_response(
    "What is machine learning?",
    return_sources=True,
    return_confidence=True
)

print(f"Response: {response['response']}")
print(f"Confidence: {response['confidence']:.3f}")
```

### Advanced Usage

```python
from rag_agent.agent import RAGAgent
from rag_agent.vectorstore import AdvancedVectorStore

# Create advanced vector store with custom settings
vectorstore = AdvancedVectorStore(
    chunk_size=300,
    chunk_overlap=50,
    use_hybrid_search=True,
    similarity_metrics=['cosine', 'euclidean']
)

# Initialize agent with advanced features
agent = RAGAgent(
    max_context_length=1500,
    top_k_retrieval=8,
    response_temperature=0.6,
    enable_query_expansion=True,
    enable_reranking=True
)

# Add documents with metadata
documents = ["Your documents here..."]
metadata = [{"category": "algorithms", "difficulty": "intermediate"}]
agent.ingest_documents(documents, metadata=metadata)

# Generate response with context filtering
response = agent.generate_response(
    "Complex query here...",
    context_filter={"category": "algorithms"},
    return_sources=True,
    return_confidence=True
)
```

## 🏗️ Architecture

### Core Components

1. **RAGAgent**: Main orchestrator class that coordinates all components
2. **AdvancedVectorStore**: Handles document storage, chunking, and retrieval
3. **ResponseGenerator**: Generates contextual responses using multiple strategies
4. **QueryProcessor**: Analyzes and expands queries for better retrieval
5. **RAGEvaluator**: Provides comprehensive evaluation metrics
6. **DocumentClassifier**: Classifies documents for improved retrieval

### Search Strategies

- **Hybrid Search**: Combines dense and sparse vector search
- **Dense Search**: Uses SVD-reduced TF-IDF vectors with cosine/euclidean similarity
- **Sparse Search**: Traditional TF-IDF with cosine similarity
- **Ensemble Search**: Aggregates results from multiple similarity metrics

### Response Types

- **Factual**: Direct, fact-based responses
- **Explanatory**: Detailed explanations with context
- **Analytical**: In-depth analysis with reasoning
- **Comparative**: Side-by-side comparisons
- **Default**: Balanced general responses

## 📊 Evaluation Metrics

The system provides comprehensive evaluation across multiple dimensions:

### Retrieval Metrics
- Average retrieval scores
- Score variance and distribution
- Coverage score (query term coverage)
- Diversity score (document diversity)

### Relevance Metrics
- Query-response relevance
- Document-response relevance
- Context utilization score

### Quality Metrics
- Readability score
- Coherence score
- Completeness score
- Consistency score

### Ground Truth Metrics (when available)
- Semantic similarity
- BLEU-like score
- Factual overlap

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python tests/test_rag_agent.py

# Run with coverage
python -m pytest tests/ --cov=rag_agent --cov-report=html
```

## 📚 Examples

### Basic Example
```bash
python examples/basic_usage.py
```

### Advanced Features Demo
```bash
python examples/advanced_usage.py
```

The advanced example demonstrates:
- Hybrid search capabilities
- Query processing and expansion
- Different response generation strategies
- Comprehensive evaluation metrics
- End-to-end workflow

## 🔧 Configuration

### RAGAgent Parameters

```python
agent = RAGAgent(
    max_context_length=4000,      # Maximum context length for responses
    top_k_retrieval=10,           # Number of documents to retrieve
    response_temperature=0.7,     # Response generation temperature
    enable_query_expansion=True,  # Enable query expansion
    enable_reranking=True         # Enable document reranking
)
```

### AdvancedVectorStore Parameters

```python
vectorstore = AdvancedVectorStore(
    chunk_size=500,               # Size of text chunks
    chunk_overlap=50,             # Overlap between chunks
    use_hybrid_search=True,       # Enable hybrid search
    svd_components=100,           # SVD components for dense vectors
    similarity_metrics=['cosine'] # Similarity metrics to use
)
```

## 📈 Performance Monitoring

The system includes built-in performance monitoring:

```python
# Get agent statistics
stats = agent.get_agent_stats()
print(f"Documents: {stats['num_documents']}")
print(f"Chunks: {stats['vectorstore_stats']['num_chunks']}")

# Get evaluation summary
evaluator = agent.evaluator
summary = evaluator.get_performance_summary()
print(f"Average confidence: {summary['current_metrics']['avg_confidence']:.3f}")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install

# Run tests
python -m pytest tests/

# Format code
black rag_agent/ tests/ examples/

# Lint code
flake8 rag_agent/ tests/ examples/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [scikit-learn](https://scikit-learn.org/) for machine learning capabilities
- Uses [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/) for numerical computing
- Inspired by modern RAG architectures and best practices

## 📞 Support

For questions, issues, or contributions:

1. Check the [Issues](../../issues) page for existing problems
2. Create a new issue with detailed description
3. Join our community discussions

## 🗺️ Roadmap

- [ ] Integration with external embedding models (OpenAI, Hugging Face)
- [ ] Support for multiple document formats (PDF, DOCX, etc.)
- [ ] Web interface for interactive querying
- [ ] Advanced caching mechanisms
- [ ] Distributed processing support
- [ ] Integration with vector databases (Pinecone, Weaviate)

---

**Built with ❤️ for the AI community**
