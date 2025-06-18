import numpy as np
import re
from typing import List, Dict, Any, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import logging

logger = logging.getLogger(__name__)


class AdvancedVectorStore:
    """
    Advanced vector storage with semantic chunking, hybrid search,
    and multiple similarity metrics.
    """

    def __init__(self,
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 use_hybrid_search: bool = True,
                 svd_components: int = 100,
                 similarity_metrics: List[str] = None):
        """
        Initialize advanced vector store.

        Args:
            chunk_size: Size of text chunks for semantic chunking
            chunk_overlap: Overlap between chunks
            use_hybrid_search: Whether to use hybrid dense+sparse search
            svd_components: Number of SVD components for dimensionality reduction
            similarity_metrics: List of similarity metrics to use
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_hybrid_search = use_hybrid_search
        self.svd_components = svd_components

        # Default similarity metrics
        if similarity_metrics is None:
            similarity_metrics = ['cosine', 'euclidean']
        self.similarity_metrics = similarity_metrics

        # Vectorizers for hybrid search
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.count_vectorizer = CountVectorizer(
            max_features=5000,
            ngram_range=(1, 1),
            stop_words='english'
        )

        # SVD for dimensionality reduction
        self.svd = TruncatedSVD(n_components=svd_components, random_state=42)

        # Storage
        self.documents = []
        self.chunks = []
        self.chunk_metadata = []
        self.tfidf_vectors = None
        self.count_vectors = None
        self.dense_vectors = None
        self.is_fitted = False

        logger.info("Advanced VectorStore initialized")

    def add_documents(self, documents: List[str], metadata: Optional[List[Dict[str, Any]]] = None):
        """
        Add documents with advanced processing including semantic chunking.

        Args:
            documents: List of document texts
            metadata: Optional metadata for each document
        """
        logger.info(f"Adding {len(documents)} documents with advanced processing")

        # Process each document
        for i, doc in enumerate(documents):
            doc_metadata = metadata[i] if metadata and i < len(metadata) else {}

            # Add original document
            self.documents.append(doc)

            # Create semantic chunks
            chunks = self._create_semantic_chunks(doc)

            # Add chunks with metadata
            for j, chunk in enumerate(chunks):
                self.chunks.append(chunk)
                chunk_meta = {
                    'document_index': len(self.documents) - 1,
                    'chunk_index': j,
                    'total_chunks': len(chunks),
                    'chunk_type': 'semantic',
                    **doc_metadata
                }
                self.chunk_metadata.append(chunk_meta)

        # Generate embeddings
        self._generate_embeddings()

        logger.info(f"Added {len(self.chunks)} chunks from {len(documents)} documents")

    def _create_semantic_chunks(self, text: str) -> List[str]:
        """
        Create semantic chunks from text using intelligent splitting.

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks
        """
        # Split by sentences first
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                current_chunk += sentence + ". "
            else:
                # Save current chunk if it's not empty
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                # Start new chunk
                current_chunk = sentence + ". "

        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Handle overlap if specified
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._add_chunk_overlap(chunks)

        return chunks if chunks else [text]  # Fallback to original text

    def _add_chunk_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between consecutive chunks."""
        overlapped_chunks = []

        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                # Get overlap from previous chunk
                prev_chunk = chunks[i-1]
                overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk

                # Combine with current chunk
                overlapped_chunk = overlap_text + " " + chunk
                overlapped_chunks.append(overlapped_chunk)

        return overlapped_chunks

    def _generate_embeddings(self):
        """Generate multiple types of embeddings for hybrid search."""
        if not self.chunks:
            return

        logger.info("Generating embeddings for chunks")

        # TF-IDF vectors (sparse)
        self.tfidf_vectors = self.tfidf_vectorizer.fit_transform(self.chunks)

        # Count vectors (sparse)
        self.count_vectors = self.count_vectorizer.fit_transform(self.chunks)

        # Dense vectors using SVD on TF-IDF
        # Adjust SVD components if necessary
        n_features = self.tfidf_vectors.shape[1]
        n_components = min(self.svd_components, n_features - 1, self.tfidf_vectors.shape[0] - 1)

        if n_components != self.svd_components:
            logger.info(f"Adjusting SVD components from {self.svd_components} to {n_components}")
            self.svd.n_components = n_components

        if n_components > 0:
            self.dense_vectors = self.svd.fit_transform(self.tfidf_vectors)
            self.dense_vectors = normalize(self.dense_vectors, norm='l2')
        else:
            # Fallback to normalized TF-IDF if SVD not possible
            logger.warning("SVD not possible, using normalized TF-IDF vectors")
            self.dense_vectors = normalize(self.tfidf_vectors.toarray(), norm='l2')

        self.is_fitted = True
        logger.info("Embeddings generated successfully")

    def similarity_search(self,
                         query: str,
                         top_k: int = 5,
                         search_type: str = 'hybrid',
                         similarity_metric: str = 'cosine') -> List[Dict[str, Any]]:
        """
        Advanced similarity search with multiple options.

        Args:
            query: Search query
            top_k: Number of results to return
            search_type: 'hybrid', 'dense', 'sparse', or 'ensemble'
            similarity_metric: Similarity metric to use

        Returns:
            List of search results with scores and metadata
        """
        if not self.is_fitted:
            raise ValueError("VectorStore must be fitted before searching")

        if search_type == 'hybrid':
            return self._hybrid_search(query, top_k, similarity_metric)
        elif search_type == 'dense':
            return self._dense_search(query, top_k, similarity_metric)
        elif search_type == 'sparse':
            return self._sparse_search(query, top_k, similarity_metric)
        elif search_type == 'ensemble':
            return self._ensemble_search(query, top_k)
        else:
            raise ValueError(f"Unknown search type: {search_type}")

    def _hybrid_search(self, query: str, top_k: int, similarity_metric: str) -> List[Dict[str, Any]]:
        """Perform hybrid dense + sparse search."""
        # Get results from both dense and sparse search
        dense_results = self._dense_search(query, top_k * 2, similarity_metric)
        sparse_results = self._sparse_search(query, top_k * 2, similarity_metric)

        # Combine and rerank results
        combined_scores = {}

        # Add dense scores (weight: 0.6)
        for result in dense_results:
            idx = result['index']
            combined_scores[idx] = combined_scores.get(idx, 0) + 0.6 * result['score']

        # Add sparse scores (weight: 0.4)
        for result in sparse_results:
            idx = result['index']
            combined_scores[idx] = combined_scores.get(idx, 0) + 0.4 * result['score']

        # Sort by combined score and return top_k
        sorted_indices = sorted(combined_scores.keys(),
                              key=lambda x: combined_scores[x],
                              reverse=True)[:top_k]

        results = []
        for idx in sorted_indices:
            results.append({
                'document': self.chunks[idx],
                'score': combined_scores[idx],
                'index': idx,
                'metadata': self.chunk_metadata[idx],
                'search_type': 'hybrid'
            })

        return results

    def _dense_search(self, query: str, top_k: int, similarity_metric: str) -> List[Dict[str, Any]]:
        """Perform dense vector search using SVD embeddings."""
        # Transform query to dense vector
        query_tfidf = self.tfidf_vectorizer.transform([query])
        query_dense = self.svd.transform(query_tfidf)
        query_dense = normalize(query_dense, norm='l2')

        # Calculate similarities
        if similarity_metric == 'cosine':
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_dense, self.dense_vectors).flatten()
        elif similarity_metric == 'euclidean':
            from sklearn.metrics.pairwise import euclidean_distances
            distances = euclidean_distances(query_dense, self.dense_vectors).flatten()
            # Convert distances to similarities (higher is better)
            similarities = 1 / (1 + distances)
        else:
            raise ValueError(f"Unknown similarity metric: {similarity_metric}")

        # Get top results
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                'document': self.chunks[idx],
                'score': similarities[idx],
                'index': idx,
                'metadata': self.chunk_metadata[idx],
                'search_type': 'dense'
            })

        return results

    def _sparse_search(self, query: str, top_k: int, similarity_metric: str) -> List[Dict[str, Any]]:
        """Perform sparse vector search using TF-IDF."""
        # Transform query to sparse vector
        query_vector = self.tfidf_vectorizer.transform([query])

        # Calculate similarities
        if similarity_metric == 'cosine':
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_vector, self.tfidf_vectors).flatten()
        else:
            # For sparse vectors, cosine is most appropriate
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_vector, self.tfidf_vectors).flatten()

        # Get top results
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                'document': self.chunks[idx],
                'score': similarities[idx],
                'index': idx,
                'metadata': self.chunk_metadata[idx],
                'search_type': 'sparse'
            })

        return results

    def _ensemble_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform ensemble search using multiple similarity metrics."""
        all_results = []

        # Get results from different metrics
        for metric in self.similarity_metrics:
            try:
                dense_results = self._dense_search(query, top_k, metric)
                sparse_results = self._sparse_search(query, top_k, metric)
                all_results.extend(dense_results)
                all_results.extend(sparse_results)
            except ValueError:
                continue  # Skip unsupported metrics

        # Aggregate scores by document index
        score_aggregation = {}
        for result in all_results:
            idx = result['index']
            if idx not in score_aggregation:
                score_aggregation[idx] = {
                    'scores': [],
                    'document': result['document'],
                    'metadata': result['metadata']
                }
            score_aggregation[idx]['scores'].append(result['score'])

        # Calculate ensemble scores (mean)
        ensemble_results = []
        for idx, data in score_aggregation.items():
            import numpy as np
            ensemble_score = np.mean(data['scores'])
            ensemble_results.append({
                'document': data['document'],
                'score': ensemble_score,
                'index': idx,
                'metadata': data['metadata'],
                'search_type': 'ensemble'
            })

        # Sort and return top_k
        ensemble_results.sort(key=lambda x: x['score'], reverse=True)
        return ensemble_results[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the vector store."""
        return {
            'num_documents': len(self.documents),
            'num_chunks': len(self.chunks),
            'avg_chunks_per_doc': len(self.chunks) / len(self.documents) if self.documents else 0,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'use_hybrid_search': self.use_hybrid_search,
            'svd_components': self.svd_components,
            'similarity_metrics': self.similarity_metrics,
            'is_fitted': self.is_fitted,
            'tfidf_features': self.tfidf_vectors.shape[1] if self.tfidf_vectors is not None else 0,
            'dense_dimensions': self.dense_vectors.shape[1] if self.dense_vectors is not None else 0
        }

    def get_document_by_chunk_index(self, chunk_index: int) -> Dict[str, Any]:
        """Get the original document for a given chunk index."""
        if chunk_index >= len(self.chunk_metadata):
            raise IndexError("Chunk index out of range")

        metadata = self.chunk_metadata[chunk_index]
        doc_index = metadata['document_index']

        return {
            'document': self.documents[doc_index],
            'chunk': self.chunks[chunk_index],
            'metadata': metadata
        }


# Maintain backward compatibility
class VectorStore(AdvancedVectorStore):
    """Backward compatible VectorStore class."""

    def __init__(self):
        super().__init__(
            chunk_size=1000,  # Larger chunks for backward compatibility
            chunk_overlap=0,   # No overlap for backward compatibility
            use_hybrid_search=False,  # Simple search for backward compatibility
            similarity_metrics=['cosine']
        )