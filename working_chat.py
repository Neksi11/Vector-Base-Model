"""
Working Chat Interface - Loads documents from my_documents.txt
"""

import sys
sys.path.append('.')

from rag_agent.agent import RAGAgent

def main():
    print("🚀 RAG Chat System - Working Version")
    print("=" * 50)
    
    # Load documents from file
    print("📚 Loading documents from my_documents.txt...")
    try:
        with open('my_documents.txt', 'r', encoding='utf-8') as f:
            documents = [line.strip() for line in f if line.strip()]
        print(f"✅ Loaded {len(documents)} documents")
    except FileNotFoundError:
        print("❌ my_documents.txt not found! Please create this file with your documents.")
        return
    
    if not documents:
        print("❌ No documents found in my_documents.txt")
        return
    
    # Initialize agent WITHOUT classifier (to avoid the pipeline error)
    print("🔧 Initializing RAG agent...")
    agent = RAGAgent(
        max_context_length=1500,
        top_k_retrieval=5,
        enable_query_expansion=False,  # Disable to avoid classifier issues
        enable_reranking=False         # Disable to avoid classifier issues
    )
    
    # Ingest documents WITHOUT labels (to avoid classifier training)
    agent.ingest_documents(documents)
    print("✅ Agent ready!")
    
    print("\n" + "=" * 50)
    print("💬 Chat with your documents!")
    print("Commands: 'quit' to exit, 'help' for help")
    print("=" * 50)
    
    while True:
        try:
            question = input("\n🤔 You: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if question.lower() == 'help':
                print("\n📖 Available commands:")
                print("  • Ask any question about your documents")
                print("  • 'quit' - Exit the chat")
                print("  • 'help' - Show this help")
                continue
            
            if not question:
                continue
            
            print("🤖 RAG Agent: Thinking...")
            
            # Get response
            result = agent.generate_response(
                question,
                return_sources=True,
                return_confidence=True
            )
            
            # Display response
            print(f"\n🤖 RAG Agent: {result['response']}")
            print(f"\n📊 Confidence: {result['confidence']:.1%}")
            
            # Show sources
            if result.get('sources'):
                print(f"\n📚 Sources ({len(result['sources'])}):")
                for i, source in enumerate(result['sources'][:3], 1):
                    print(f"  {i}. [{source['score']:.2f}] {source['document'][:80]}...")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try a different question.")

if __name__ == "__main__":
    main()
