"""
Advanced Vector-Based RAG Agent Package

This package provides a sophisticated Retrieval Augmented Generation system
with advanced response generation capabilities.
"""

from .agent import RAGAgent
from .vectorstore import VectorStore
from .classifier import DocumentClassifier
from .response_generator import ResponseGenerator
from .query_processor import QueryProcessor
from .evaluator import RAGEvaluator

__version__ = "1.0.0"
__all__ = [
    "RAGAgent",
    "VectorStore", 
    "DocumentClassifier",
    "ResponseGenerator",
    "QueryProcessor",
    "RAGEvaluator"
]
