from rag_agent.agent import RAGAgent

def main():
    # Create sample documents
    documents = [
        "Random forests are an ensemble learning method for classification and regression.",
        "Scikit-learn provides tools for machine learning and statistical modeling.",
        "Vector embeddings represent text as numerical vectors in a high-dimensional space.",
        "Retrieval Augmented Generation combines information retrieval with text generation.",
        "Classification algorithms assign categories to input data based on training examples."
    ]
    
    # Create labels for documents
    labels = ["algorithm", "library", "embedding", "technique", "algorithm"]
    
    # Initialize the RAG agent
    agent = RAGAgent()
    
    # Ingest documents and train classifier
    agent.ingest_documents(documents, labels)
    
    # Process a query
    query = "How do vector embeddings work?"
    response = agent.generate_response(query)
    
    print(f"Query: {query}")
    print(f"Response: {response}")
    
    # Process another query
    query = "Tell me about random forests"
    response = agent.generate_response(query)
    
    print(f"\nQuery: {query}")
    print(f"Response: {response}")

if __name__ == "__main__":
    main()