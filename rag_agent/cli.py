"""
Command Line Interface for the Advanced RAG Agent.
"""

import argparse
import json
import yaml
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

from .agent import RAGAgent
from .vectorstore import AdvancedVectorStore


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}


def load_documents_from_file(file_path: str) -> List[str]:
    """Load documents from a text file (one document per line)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            documents = [line.strip() for line in f if line.strip()]
        return documents
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []


def load_documents_from_json(file_path: str) -> tuple[List[str], List[str], List[Dict[str, Any]]]:
    """Load documents, labels, and metadata from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = data.get('documents', [])
        labels = data.get('labels', [])
        metadata = data.get('metadata', [])
        
        return documents, labels, metadata
    except Exception as e:
        print(f"Error loading JSON data: {e}")
        return [], [], []


def interactive_mode(agent: RAGAgent):
    """Run the agent in interactive mode."""
    print("Advanced RAG Agent - Interactive Mode")
    print("Type 'quit' to exit, 'help' for commands")
    print("-" * 50)
    
    while True:
        try:
            query = input("\nQuery: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if query.lower() == 'help':
                print("\nAvailable commands:")
                print("  help - Show this help message")
                print("  stats - Show agent statistics")
                print("  quit/exit/q - Exit the program")
                print("  Any other text - Process as a query")
                continue
            
            if query.lower() == 'stats':
                stats = agent.get_agent_stats()
                print(f"\nAgent Statistics:")
                print(f"  Documents: {stats['num_documents']}")
                print(f"  Trained: {stats['is_trained']}")
                if 'vectorstore_stats' in stats:
                    vs_stats = stats['vectorstore_stats']
                    print(f"  Chunks: {vs_stats.get('num_chunks', 'N/A')}")
                    print(f"  Avg chunks per doc: {vs_stats.get('avg_chunks_per_doc', 'N/A'):.1f}")
                continue
            
            if not query:
                continue
            
            print("Processing...")
            start_time = time.time()
            
            response = agent.generate_response(
                query,
                return_sources=True,
                return_confidence=True
            )
            
            response_time = time.time() - start_time
            
            print(f"\nResponse: {response['response']}")
            print(f"Confidence: {response['confidence']:.3f}")
            print(f"Response time: {response_time:.3f}s")
            
            if response.get('sources'):
                print(f"\nSources ({len(response['sources'])}):")
                for i, source in enumerate(response['sources'][:3], 1):
                    print(f"  {i}. Score: {source['score']:.3f}")
                    print(f"     Text: {source['document'][:100]}...")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def batch_mode(agent: RAGAgent, queries_file: str, output_file: str = None):
    """Process queries in batch mode."""
    try:
        with open(queries_file, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error loading queries: {e}")
        return
    
    results = []
    
    print(f"Processing {len(queries)} queries...")
    
    for i, query in enumerate(queries, 1):
        print(f"Processing query {i}/{len(queries)}: {query[:50]}...")
        
        try:
            start_time = time.time()
            response = agent.generate_response(
                query,
                return_sources=True,
                return_confidence=True
            )
            response_time = time.time() - start_time
            
            result = {
                'query': query,
                'response': response['response'],
                'confidence': response['confidence'],
                'response_time': response_time,
                'sources_count': len(response.get('sources', [])),
                'evaluation': response.get('evaluation', {})
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing query: {e}")
            results.append({
                'query': query,
                'error': str(e)
            })
    
    # Save results
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    # Print summary
    successful = len([r for r in results if 'error' not in r])
    avg_confidence = sum(r.get('confidence', 0) for r in results if 'confidence' in r) / max(successful, 1)
    avg_time = sum(r.get('response_time', 0) for r in results if 'response_time' in r) / max(successful, 1)
    
    print(f"\nBatch Processing Summary:")
    print(f"  Total queries: {len(queries)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {len(queries) - successful}")
    print(f"  Average confidence: {avg_confidence:.3f}")
    print(f"  Average response time: {avg_time:.3f}s")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Advanced RAG Agent Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with default config
  python -m rag_agent.cli --interactive --documents data.txt
  
  # Batch processing
  python -m rag_agent.cli --batch queries.txt --documents data.json --output results.json
  
  # Custom configuration
  python -m rag_agent.cli --interactive --config custom_config.yaml --documents data.json
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/default_config.yaml',
        help='Path to configuration file (default: config/default_config.yaml)'
    )
    
    parser.add_argument(
        '--documents', '-d',
        type=str,
        required=True,
        help='Path to documents file (.txt for plain text, .json for structured data)'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--batch', '-b',
        type=str,
        help='Run in batch mode with queries from file'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file for batch mode results'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.interactive and not args.batch:
        print("Error: Must specify either --interactive or --batch mode")
        sys.exit(1)
    
    if args.interactive and args.batch:
        print("Error: Cannot use both --interactive and --batch modes")
        sys.exit(1)
    
    # Load configuration
    config = {}
    if Path(args.config).exists():
        config = load_config(args.config)
        if args.verbose:
            print(f"Loaded configuration from {args.config}")
    else:
        print(f"Warning: Config file {args.config} not found, using defaults")
    
    # Initialize agent with config
    agent_config = config.get('rag_agent', {})
    agent = RAGAgent(
        max_context_length=agent_config.get('max_context_length', 4000),
        top_k_retrieval=agent_config.get('top_k_retrieval', 10),
        response_temperature=agent_config.get('response_temperature', 0.7),
        enable_query_expansion=agent_config.get('enable_query_expansion', True),
        enable_reranking=agent_config.get('enable_reranking', True)
    )
    
    # Load documents
    print(f"Loading documents from {args.documents}...")
    
    if args.documents.endswith('.json'):
        documents, labels, metadata = load_documents_from_json(args.documents)
    else:
        documents = load_documents_from_file(args.documents)
        labels = None
        metadata = None
    
    if not documents:
        print("Error: No documents loaded")
        sys.exit(1)
    
    print(f"Loaded {len(documents)} documents")
    
    # Ingest documents
    agent.ingest_documents(documents, labels, metadata)
    print("Documents ingested successfully")
    
    # Run in specified mode
    if args.interactive:
        interactive_mode(agent)
    elif args.batch:
        batch_mode(agent, args.batch, args.output)


if __name__ == '__main__':
    main()
