"""Result processing for retrieval results."""
import re
import asyncio
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict
import hashlib
from datetime import datetime

from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


class ResultProcessor:
    """Process and enhance retrieval results."""
    
    def __init__(
        self,
        deduplication_threshold: float = 0.95,
        clustering_threshold: float = 0.8,
        highlight_query: bool = True,
        max_snippet_length: int = 200,
        snippet_overlap: int = 50
    ):
        self.deduplication_threshold = deduplication_threshold
        self.clustering_threshold = clustering_threshold
        self.highlight_query = highlight_query
        self.max_snippet_length = max_snippet_length
        self.snippet_overlap = snippet_overlap
        self.vectorizer = TfidfVectorizer(max_features=1000)
        
    def process_results(
        self,
        documents: List[Document],
        query: str,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Process retrieval results with all enhancements."""
        if not documents:
            return []
            
        # 1. Apply metadata filtering if provided
        if metadata_filter:
            documents = self._filter_by_metadata(documents, metadata_filter)
            
        # 2. Deduplicate results
        documents = self._deduplicate_documents(documents)
        
        # 3. Cluster similar documents
        clusters = self._cluster_documents(documents)
        
        # 4. Enhance with snippets and highlights
        documents = self._enhance_documents(documents, query, clusters)
        
        # 5. Format citations
        documents = self._format_citations(documents)
        
        return documents
        
    def _filter_by_metadata(
        self,
        documents: List[Document],
        metadata_filter: Dict[str, Any]
    ) -> List[Document]:
        """Filter documents by metadata criteria."""
        filtered = []
        for doc in documents:
            match = True
            for key, value in metadata_filter.items():
                if key not in doc.metadata:
                    match = False
                    break
                    
                doc_value = doc.metadata[key]
                if isinstance(value, list):
                    # Match if doc value is in the list
                    if doc_value not in value:
                        match = False
                        break
                elif callable(value):
                    # Custom filter function
                    if not value(doc_value):
                        match = False
                        break
                else:
                    # Direct comparison
                    if doc_value != value:
                        match = False
                        break
                        
            if match:
                filtered.append(doc)
                
        return filtered
        
    def _deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        """Remove duplicate documents based on content similarity."""
        if len(documents) <= 1:
            return documents
            
        # Calculate content hashes
        content_hashes = []
        for doc in documents:
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            content_hashes.append(content_hash)
            
        # Build TF-IDF vectors for similarity comparison
        try:
            vectors = self.vectorizer.fit_transform([doc.page_content for doc in documents])
            similarity_matrix = cosine_similarity(vectors)
        except:
            # Fallback to simple hash-based deduplication
            unique_docs = []
            seen_hashes = set()
            for i, doc in enumerate(documents):
                if content_hashes[i] not in seen_hashes:
                    seen_hashes.add(content_hashes[i])
                    unique_docs.append(doc)
            return unique_docs
            
        # Keep track of documents to keep
        keep_indices = set(range(len(documents)))
        
        for i in range(len(documents)):
            if i not in keep_indices:
                continue
                
            for j in range(i + 1, len(documents)):
                if j not in keep_indices:
                    continue
                    
                # Check similarity
                if similarity_matrix[i, j] >= self.deduplication_threshold:
                    # Keep the document with higher score or more metadata
                    score_i = documents[i].metadata.get("score", 0)
                    score_j = documents[j].metadata.get("score", 0)
                    
                    if score_i >= score_j:
                        keep_indices.discard(j)
                    else:
                        keep_indices.discard(i)
                        break
                        
        return [documents[i] for i in sorted(keep_indices)]
        
    def _cluster_documents(self, documents: List[Document]) -> Dict[int, List[int]]:
        """Cluster similar documents together."""
        if len(documents) <= 1:
            return {0: [0]}
            
        try:
            # Build TF-IDF vectors
            vectors = self.vectorizer.fit_transform([doc.page_content for doc in documents])
            similarity_matrix = cosine_similarity(vectors)
            
            # Simple hierarchical clustering
            clusters = defaultdict(list)
            assigned = set()
            cluster_id = 0
            
            for i in range(len(documents)):
                if i in assigned:
                    continue
                    
                # Start new cluster
                clusters[cluster_id].append(i)
                assigned.add(i)
                
                # Find similar documents
                for j in range(i + 1, len(documents)):
                    if j in assigned:
                        continue
                        
                    if similarity_matrix[i, j] >= self.clustering_threshold:
                        clusters[cluster_id].append(j)
                        assigned.add(j)
                        
                cluster_id += 1
                
            return dict(clusters)
            
        except:
            # Fallback: each document in its own cluster
            return {i: [i] for i in range(len(documents))}
            
    def _enhance_documents(
        self,
        documents: List[Document],
        query: str,
        clusters: Dict[int, List[int]]
    ) -> List[Document]:
        """Enhance documents with snippets and highlights."""
        enhanced_docs = []
        
        # Reverse mapping: doc_index -> cluster_id
        doc_to_cluster = {}
        for cluster_id, doc_indices in clusters.items():
            for doc_idx in doc_indices:
                doc_to_cluster[doc_idx] = cluster_id
                
        for i, doc in enumerate(documents):
            # Create snippet
            snippet = self._create_snippet(doc.page_content, query)
            
            # Highlight query terms if enabled
            if self.highlight_query:
                snippet = self._highlight_terms(snippet, query)
                
            # Add cluster information
            cluster_id = doc_to_cluster.get(i, -1)
            cluster_size = len(clusters.get(cluster_id, []))
            
            # Update metadata
            enhanced_metadata = doc.metadata.copy()
            enhanced_metadata.update({
                "snippet": snippet,
                "cluster_id": cluster_id,
                "cluster_size": cluster_size,
                "processing_timestamp": datetime.utcnow().isoformat()
            })
            
            enhanced_doc = Document(
                page_content=doc.page_content,
                metadata=enhanced_metadata
            )
            enhanced_docs.append(enhanced_doc)
            
        return enhanced_docs
        
    def _create_snippet(self, content: str, query: str) -> str:
        """Create a relevant snippet from the content."""
        # Find query terms in content
        query_terms = query.lower().split()
        content_lower = content.lower()
        
        # Find best matching position
        best_pos = 0
        best_score = 0
        
        for i in range(len(content)):
            score = 0
            for term in query_terms:
                # Check if term appears near this position
                window = content_lower[max(0, i-50):i+50]
                if term in window:
                    score += 1
                    
            if score > best_score:
                best_score = score
                best_pos = i
                
        # Extract snippet around best position
        start = max(0, best_pos - self.max_snippet_length // 2)
        end = min(len(content), start + self.max_snippet_length)
        
        # Adjust to word boundaries
        if start > 0:
            while start < len(content) and content[start] not in ' \n\t':
                start += 1
        if end < len(content):
            while end > 0 and content[end-1] not in ' \n\t':
                end -= 1
                
        snippet = content[start:end].strip()
        
        # Add ellipsis if needed
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."
            
        return snippet
        
    def _highlight_terms(self, text: str, query: str) -> str:
        """Highlight query terms in text."""
        # Split query into terms
        query_terms = set(query.lower().split())
        
        # Create regex pattern for highlighting
        pattern = r'\b(' + '|'.join(re.escape(term) for term in query_terms) + r')\b'
        
        # Replace with highlighted version
        def replace_func(match):
            return f"**{match.group(0)}**"
            
        highlighted = re.sub(pattern, replace_func, text, flags=re.IGNORECASE)
        return highlighted
        
    def _format_citations(self, documents: List[Document]) -> List[Document]:
        """Format citations for documents."""
        for i, doc in enumerate(documents):
            # Generate citation ID
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", None)
            
            # Create citation format
            if page:
                citation = f"[{i+1}] {source}, p. {page}"
            else:
                citation = f"[{i+1}] {source}"
                
            doc.metadata["citation"] = citation
            doc.metadata["citation_id"] = i + 1
            
        return documents


class StreamingResultProcessor(ResultProcessor):
    """Result processor with streaming support."""
    
    async def process_results_stream(
        self,
        documents: List[Document],
        query: str,
        metadata_filter: Optional[Dict[str, Any]] = None
    ):
        """Process results and yield them as they're ready."""
        if not documents:
            return
            
        # Apply metadata filtering if provided
        if metadata_filter:
            documents = self._filter_by_metadata(documents, metadata_filter)
            
        # Deduplicate first (important for streaming)
        documents = self._deduplicate_documents(documents)
        
        # Process and yield documents one by one
        processed_count = 0
        for doc in documents:
            # Create snippet
            snippet = self._create_snippet(doc.page_content, query)
            
            # Highlight query terms
            if self.highlight_query:
                snippet = self._highlight_terms(snippet, query)
                
            # Format citation
            processed_count += 1
            citation = f"[{processed_count}] {doc.metadata.get('source', 'Unknown')}"
            
            # Update metadata
            enhanced_metadata = doc.metadata.copy()
            enhanced_metadata.update({
                "snippet": snippet,
                "citation": citation,
                "citation_id": processed_count,
                "processing_timestamp": datetime.utcnow().isoformat()
            })
            
            enhanced_doc = Document(
                page_content=doc.page_content,
                metadata=enhanced_metadata
            )
            
            yield enhanced_doc
            
            # Small delay to prevent overwhelming the client
            await asyncio.sleep(0.01)