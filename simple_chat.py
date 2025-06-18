"""
Simple Chat Interface for RAG Agent
Run this script to start chatting with your documents!
"""

import sys
sys.path.append('.')

from rag_agent.agent import RAGAgent
import time

def main():
    print("🚀 Starting Advanced RAG Chat System...")
    print("=" * 50)
    
    # Sample documents - Replace with your own!
    documents = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make predictions or decisions.",
        
        "Python is a high-level programming language known for its simplicity and readability. It's widely used in data science, machine learning, web development, and automation. Python has extensive libraries like NumPy, Pandas, and Scikit-learn.",
        
        "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information. Deep learning uses neural networks with multiple layers to solve complex problems.",
        
        "Data science combines statistics, programming, and domain expertise to extract insights from data. It involves data collection, cleaning, analysis, visualization, and interpretation to solve real-world problems.",
        
        "Artificial intelligence (AI) refers to the simulation of human intelligence in machines. AI systems can perform tasks that typically require human intelligence, such as visual perception, speech recognition, and decision-making.",
        
        "Scikit-learn is a popular machine learning library for Python. It provides simple and efficient tools for data mining and data analysis. It includes algorithms for classification, regression, clustering, and dimensionality reduction."
    ]
    
    # Optional labels for better classification
    labels = ["machine_learning", "python", "neural_networks", "data_science", "artificial_intelligence", "scikit_learn"]
    
    # Initialize the RAG agent
    print("🔧 Initializing RAG Agent...")
    agent = RAGAgent(
        max_context_length=2000,
        top_k_retrieval=5,
        enable_query_expansion=True,
        enable_reranking=True
    )
    
    # Train the agent with documents
    print("📚 Training agent with documents...")
    agent.ingest_documents(documents, labels)
    print("✅ Agent ready!")
    
    print("\n" + "=" * 50)
    print("💬 RAG Chat Interface")
    print("Ask me anything about the documents!")
    print("Commands:")
    print("  'quit' or 'exit' - Exit the chat")
    print("  'help' - Show this help")
    print("  'stats' - Show system statistics")
    print("=" * 50)
    
    while True:
        try:
            # Get user input
            user_input = input("\n🤔 You: ").strip()
            
            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye! Thanks for using the RAG system!")
                break
                
            if user_input.lower() == 'help':
                print("\n📖 Help:")
                print("Ask questions like:")
                print("  • 'What is machine learning?'")
                print("  • 'How does Python help in data science?'")
                print("  • 'Explain neural networks'")
                print("  • 'Compare AI and machine learning'")
                print("  • 'What can I do with scikit-learn?'")
                continue
                
            if user_input.lower() == 'stats':
                stats = agent.get_agent_stats()
                print(f"\n📊 System Statistics:")
                print(f"  Documents: {stats['num_documents']}")
                print(f"  Trained: {stats['is_trained']}")
                if 'vectorstore_stats' in stats:
                    vs_stats = stats['vectorstore_stats']
                    print(f"  Chunks: {vs_stats.get('num_chunks', 'N/A')}")
                continue
            
            if not user_input:
                print("Please ask a question!")
                continue
            
            # Process the query
            print("🤖 RAG Agent: Thinking...")
            start_time = time.time()
            
            response = agent.generate_response(
                user_input,
                return_sources=True,
                return_confidence=True
            )
            
            response_time = time.time() - start_time
            
            # Display the response
            print(f"\n🤖 RAG Agent: {response['response']}")
            print(f"\n📊 Confidence: {response['confidence']:.1%}")
            print(f"⏱️  Response time: {response_time:.2f}s")
            
            # Show sources if available
            if response.get('sources'):
                print(f"\n📚 Sources used ({len(response['sources'])}):")
                for i, source in enumerate(response['sources'][:3], 1):
                    print(f"  {i}. [{source['score']:.2f}] {source['document'][:80]}...")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again with a different question.")

if __name__ == "__main__":
    main()
