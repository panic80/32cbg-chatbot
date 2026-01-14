"""
Content Co-occurrence Indexer - A generic approach to indexing content.

This indexer builds a graph of co-occurring terms within proximity windows,
treating all content uniformly regardless of structure (tables, text, JSON, etc).
"""

import re
import pickle
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
from pathlib import Path
import logging

from langchain_core.documents import Document

from app.core.logging import get_logger

logger = get_logger(__name__)


class CooccurrenceEdge:
    """Represents an edge between two terms with distance information."""
    
    def __init__(self):
        self.distances: Dict[int, int] = defaultdict(int)  # distance -> count
        self.documents: Set[str] = set()  # document IDs containing this edge
        self.contexts: List[Tuple[str, str]] = []  # (doc_id, surrounding_text) samples
        
    def add_occurrence(self, distance: int, doc_id: str, context: str = ""):
        """Add an occurrence of this edge."""
        self.distances[distance] += 1
        self.documents.add(doc_id)
        
        # Keep up to 5 context samples
        if len(self.contexts) < 5 and context:
            self.contexts.append((doc_id, context))
            
    def get_min_distance(self) -> int:
        """Get minimum distance between terms."""
        return min(self.distances.keys()) if self.distances else float('inf')
        
    def get_weighted_score(self) -> float:
        """Calculate a weighted score based on distances."""
        score = 0.0
        distance_weights = {
            0: 1.0,    # Same position (exact match)
            1: 0.9,    # Adjacent
            2: 0.8,    # Very close
            3: 0.7,    # Close
            4: 0.6,    # Near
            5: 0.5,    # Somewhat near
        }
        
        for distance, count in self.distances.items():
            if distance <= 5:
                weight = distance_weights.get(distance, 0.4)
            else:
                # Decay weight for larger distances
                weight = max(0.1, 0.4 * (1 / (distance - 4)))
            score += weight * count
            
        return score


class CooccurrenceIndexer:
    """Indexes content based on term co-occurrence within proximity windows."""
    
    def __init__(self, 
                 window_sizes: List[int] = None,
                 min_token_length: int = 2,
                 index_path: Optional[Path] = None):
        """
        Initialize the co-occurrence indexer.
        
        Args:
            window_sizes: List of window sizes for proximity indexing
            min_token_length: Minimum token length to index
            index_path: Path to save/load the index
        """
        self.window_sizes = window_sizes or [5, 20, 50, 100]
        self.min_token_length = min_token_length
        self.index_path = Path(index_path) if index_path else Path("cooccurrence_index")
        
        # Core data structures
        self.cooccurrence_graph: Dict[str, Dict[str, CooccurrenceEdge]] = defaultdict(dict)
        self.document_tokens: Dict[str, List[str]] = {}  # doc_id -> tokens
        self.token_positions: Dict[str, Dict[str, List[int]]] = defaultdict(dict)  # token -> {doc_id -> positions}
        self.document_metadata: Dict[str, Dict] = {}  # doc_id -> metadata
        
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into terms."""
        # Convert to lowercase and split on non-alphanumeric characters
        tokens = re.findall(r'\b[\w\$\%\.\-]+\b', text.lower())
        
        # Filter by minimum length
        tokens = [t for t in tokens if len(t) >= self.min_token_length]
        
        return tokens
        
    def add_document(self, document: Document):
        """Add a document to the index."""
        doc_id = document.metadata.get("id", hash(document.page_content))
        doc_id = str(doc_id)
        
        # Store metadata
        self.document_metadata[doc_id] = document.metadata
        
        # Tokenize content
        tokens = self.tokenize(document.page_content)
        self.document_tokens[doc_id] = tokens
        
        # Store token positions
        for i, token in enumerate(tokens):
            if doc_id not in self.token_positions[token]:
                self.token_positions[token][doc_id] = []
            self.token_positions[token][doc_id].append(i)
        
        # Build co-occurrence edges for different window sizes
        for window_size in self.window_sizes:
            self._index_cooccurrences(doc_id, tokens, window_size)
            
        logger.info(f"Indexed document {doc_id} with {len(tokens)} tokens")
        
    def _index_cooccurrences(self, doc_id: str, tokens: List[str], window_size: int):
        """Index co-occurrences within a specific window size."""
        for i, token1 in enumerate(tokens):
            # Look ahead within window
            for j in range(i + 1, min(i + window_size + 1, len(tokens))):
                token2 = tokens[j]
                distance = j - i
                
                # Skip if same token
                if token1 == token2:
                    continue
                    
                # Create bidirectional edges
                context_start = max(0, i - 2)
                context_end = min(len(tokens), j + 3)
                context = " ".join(tokens[context_start:context_end])
                
                # Add edge token1 -> token2
                if token2 not in self.cooccurrence_graph[token1]:
                    self.cooccurrence_graph[token1][token2] = CooccurrenceEdge()
                self.cooccurrence_graph[token1][token2].add_occurrence(distance, doc_id, context)
                
                # Add edge token2 -> token1
                if token1 not in self.cooccurrence_graph[token2]:
                    self.cooccurrence_graph[token2][token1] = CooccurrenceEdge()
                self.cooccurrence_graph[token2][token1].add_occurrence(distance, doc_id, context)
                
    def find_connecting_content(self, query_terms: List[str], top_k: int = 10) -> List[Tuple[str, float, Dict]]:
        """
        Find content that connects the query terms with minimal distance.
        
        Returns:
            List of (doc_id, score, metadata) tuples
        """
        query_tokens = []
        for term in query_terms:
            query_tokens.extend(self.tokenize(term))
        
        # Remove duplicates while preserving order
        query_tokens = list(dict.fromkeys(query_tokens))
        
        if not query_tokens:
            return []
            
        # Find documents containing at least one query token
        candidate_docs = set()
        for token in query_tokens:
            if token in self.token_positions:
                candidate_docs.update(self.token_positions[token].keys())
                
        if not candidate_docs:
            logger.info(f"No documents found containing query tokens: {query_tokens}")
            return []
            
        # Score each candidate document
        doc_scores = []
        
        for doc_id in candidate_docs:
            score, details = self._score_document(doc_id, query_tokens)
            if score > 0:
                doc_scores.append((doc_id, score, details))
                
        # Sort by score and return top k
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_id, score, details in doc_scores[:top_k]:
            results.append((
                doc_id,
                score,
                {
                    "metadata": self.document_metadata.get(doc_id, {}),
                    "matching_tokens": details["matching_tokens"],
                    "proximity_scores": details["proximity_scores"],
                    "sample_contexts": details["sample_contexts"]
                }
            ))
            
        return results
        
    def _score_document(self, doc_id: str, query_tokens: List[str]) -> Tuple[float, Dict]:
        """Score a document based on query token proximity."""
        doc_tokens = self.document_tokens.get(doc_id, [])
        if not doc_tokens:
            return 0.0, {}
            
        # Find which query tokens are in this document
        present_tokens = []
        token_positions = {}
        
        for token in query_tokens:
            if token in self.token_positions and doc_id in self.token_positions[token]:
                present_tokens.append(token)
                token_positions[token] = self.token_positions[token][doc_id]
                
        if not present_tokens:
            return 0.0, {}
            
        # Calculate proximity scores between present tokens
        proximity_scores = {}
        sample_contexts = []
        
        if len(present_tokens) == 1:
            # Single token match - base score on frequency
            token = present_tokens[0]
            frequency = len(token_positions[token])
            score = min(1.0 + (frequency - 1) * 0.1, 2.0)  # Cap at 2.0
            proximity_scores[f"{token}_frequency"] = frequency
            
            # Get sample context
            pos = token_positions[token][0]
            context_start = max(0, pos - 5)
            context_end = min(len(doc_tokens), pos + 6)
            context = " ".join(doc_tokens[context_start:context_end])
            sample_contexts.append(context)
            
        else:
            # Multiple tokens - calculate pairwise proximities
            total_score = 0.0
            num_pairs = 0
            
            for i, token1 in enumerate(present_tokens):
                for j in range(i + 1, len(present_tokens)):
                    token2 = present_tokens[j]
                    
                    # Calculate minimum distance between all occurrences
                    min_distance = float('inf')
                    best_pos1, best_pos2 = -1, -1
                    
                    for pos1 in token_positions[token1]:
                        for pos2 in token_positions[token2]:
                            distance = abs(pos2 - pos1)
                            if distance < min_distance:
                                min_distance = distance
                                best_pos1, best_pos2 = pos1, pos2
                                
                    # Convert distance to score
                    if min_distance < float('inf'):
                        if min_distance == 0:
                            pair_score = 2.0  # Same position
                        elif min_distance <= 5:
                            pair_score = 1.5 - (min_distance - 1) * 0.1
                        elif min_distance <= 20:
                            pair_score = 1.0 - (min_distance - 5) * 0.03
                        elif min_distance <= 50:
                            pair_score = 0.5 - (min_distance - 20) * 0.01
                        else:
                            pair_score = max(0.1, 0.2 * (50 / min_distance))
                            
                        total_score += pair_score
                        num_pairs += 1
                        proximity_scores[f"{token1}_{token2}"] = min_distance
                        
                        # Get sample context for closest pair
                        if len(sample_contexts) < 3:
                            context_start = max(0, min(best_pos1, best_pos2) - 3)
                            context_end = min(len(doc_tokens), max(best_pos1, best_pos2) + 4)
                            context = " ".join(doc_tokens[context_start:context_end])
                            sample_contexts.append(context)
                            
            score = total_score / max(1, num_pairs) if num_pairs > 0 else 0.0
            
            # Boost score if all query tokens are present
            if len(present_tokens) == len(query_tokens):
                score *= 1.5
                
        details = {
            "matching_tokens": present_tokens,
            "proximity_scores": proximity_scores,
            "sample_contexts": sample_contexts
        }
        
        return score, details
        
    def save_index(self):
        """Save the index to disk."""
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Save main graph
        with open(self.index_path / "cooccurrence_graph.pkl", "wb") as f:
            pickle.dump(dict(self.cooccurrence_graph), f)
            
        # Save auxiliary data
        with open(self.index_path / "document_tokens.pkl", "wb") as f:
            pickle.dump(self.document_tokens, f)
            
        with open(self.index_path / "token_positions.pkl", "wb") as f:
            pickle.dump(dict(self.token_positions), f)
            
        with open(self.index_path / "document_metadata.pkl", "wb") as f:
            pickle.dump(self.document_metadata, f)
            
        logger.info(f"Saved co-occurrence index to {self.index_path}")
        
    def load_index(self) -> bool:
        """Load the index from disk."""
        try:
            # Load main graph
            with open(self.index_path / "cooccurrence_graph.pkl", "rb") as f:
                self.cooccurrence_graph = defaultdict(dict, pickle.load(f))
                
            # Load auxiliary data
            with open(self.index_path / "document_tokens.pkl", "rb") as f:
                self.document_tokens = pickle.load(f)
                
            with open(self.index_path / "token_positions.pkl", "rb") as f:
                self.token_positions = defaultdict(dict, pickle.load(f))
                
            with open(self.index_path / "document_metadata.pkl", "rb") as f:
                self.document_metadata = pickle.load(f)
                
            logger.info(f"Loaded co-occurrence index from {self.index_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load co-occurrence index: {e}")
            return False
            
    def clear_index(self):
        """Clear the index."""
        self.cooccurrence_graph.clear()
        self.document_tokens.clear()
        self.token_positions.clear()
        self.document_metadata.clear()
        logger.info("Cleared co-occurrence index")