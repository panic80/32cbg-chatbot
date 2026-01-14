"""
Co-occurrence Retriever component for concept-based retrieval.

This module provides a retriever that uses co-occurrence patterns
to find documents related to concepts that frequently appear together.
"""

from typing import List, Dict, Any, Optional, Set
import logging
import json
import os
from collections import defaultdict

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field

from app.components.base import BaseComponent
from app.core.logging import get_logger

logger = get_logger(__name__)


class CooccurrenceIndex:
    """Index for tracking concept co-occurrences."""
    
    def __init__(self, index_path: str = "cooccurrence_index"):
        self.index_path = index_path
        self.cooccurrence_map: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.document_map: Dict[str, Set[str]] = defaultdict(set)
        self.concept_docs: Dict[str, Set[str]] = defaultdict(set)
        
    def add_document(self, doc_id: str, concepts: List[str]):
        """Add a document with its concepts to the index."""
        # Track which concepts appear in this document
        for concept in concepts:
            self.concept_docs[concept].add(doc_id)
            
        # Update co-occurrence counts
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:
                self.cooccurrence_map[concept1][concept2] += 1
                self.cooccurrence_map[concept2][concept1] += 1
                
        self.document_map[doc_id] = set(concepts)
        
    def get_related_concepts(self, concept: str, top_k: int = 10) -> List[tuple[str, float]]:
        """Get concepts that frequently co-occur with the given concept."""
        if concept not in self.cooccurrence_map:
            return []
            
        related = sorted(
            self.cooccurrence_map[concept].items(),
            key=lambda x: x[1],
            reverse=True
        )
        return related[:top_k]
        
    def get_documents_for_concepts(self, concepts: List[str]) -> Set[str]:
        """Get documents containing any of the given concepts."""
        doc_ids = set()
        for concept in concepts:
            doc_ids.update(self.concept_docs.get(concept, set()))
        return doc_ids
        
    def save(self):
        """Save index to disk."""
        os.makedirs(self.index_path, exist_ok=True)
        
        # Save co-occurrence map
        with open(os.path.join(self.index_path, "cooccurrence.json"), "w") as f:
            json.dump(dict(self.cooccurrence_map), f)
            
        # Save document map
        with open(os.path.join(self.index_path, "documents.json"), "w") as f:
            doc_map_serializable = {k: list(v) for k, v in self.document_map.items()}
            json.dump(doc_map_serializable, f)
            
        # Save concept-docs map
        with open(os.path.join(self.index_path, "concept_docs.json"), "w") as f:
            concept_docs_serializable = {k: list(v) for k, v in self.concept_docs.items()}
            json.dump(concept_docs_serializable, f)
            
    def load(self) -> bool:
        """Load index from disk."""
        try:
            # Load co-occurrence map
            with open(os.path.join(self.index_path, "cooccurrence.json"), "r") as f:
                data = json.load(f)
                self.cooccurrence_map = defaultdict(lambda: defaultdict(float))
                for k1, v1 in data.items():
                    for k2, v2 in v1.items():
                        self.cooccurrence_map[k1][k2] = v2
                        
            # Load document map
            with open(os.path.join(self.index_path, "documents.json"), "r") as f:
                data = json.load(f)
                self.document_map = {k: set(v) for k, v in data.items()}
                
            # Load concept-docs map
            with open(os.path.join(self.index_path, "concept_docs.json"), "r") as f:
                data = json.load(f)
                self.concept_docs = defaultdict(set)
                for k, v in data.items():
                    self.concept_docs[k] = set(v)
                    
            return True
        except Exception as e:
            logger.warning(f"Failed to load co-occurrence index: {e}")
            return False


class TravelCooccurrenceRetriever(BaseRetriever, BaseComponent):
    """
    Co-occurrence based retriever for travel documents.
    
    Uses concept co-occurrence patterns to find related documents.
    """
    
    index: CooccurrenceIndex = Field(description="Co-occurrence index")
    documents: Dict[str, Document] = Field(default_factory=dict, description="Document store")
    k: int = Field(default=10, description="Number of documents to retrieve")
    expansion_k: int = Field(default=5, description="Number of related concepts to expand")
    
    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True
    
    def __init__(
        self,
        documents: List[Document],
        index_path: str = "cooccurrence_index",
        k: int = 10,
        expansion_k: int = 5,
        **kwargs
    ):
        """
        Initialize the co-occurrence retriever.
        
        Args:
            documents: List of documents to index
            index_path: Path to store/load the index
            k: Number of documents to retrieve
            expansion_k: Number of related concepts for expansion
        """
        # Initialize BaseComponent
        BaseComponent.__init__(self, component_type="retriever", component_name="cooccurrence")
        
        # Create index
        index = CooccurrenceIndex(index_path)
        
        # Try to load existing index
        if not index.load():
            logger.info("Building new co-occurrence index...")
            # Build index from documents
            for doc in documents:
                # Extract concepts from document
                concepts = self._extract_concepts(doc)
                doc_id = doc.metadata.get("source", str(hash(doc.page_content)))
                index.add_document(doc_id, concepts)
            
            # Save index
            index.save()
            logger.info("Co-occurrence index built and saved")
        
        # Create document map
        doc_map = {}
        for doc in documents:
            doc_id = doc.metadata.get("source", str(hash(doc.page_content)))
            doc_map[doc_id] = doc
        
        # Initialize BaseRetriever with fields
        super().__init__(
            index=index,
            documents=doc_map,
            k=k,
            expansion_k=expansion_k,
            **kwargs
        )
        
        logger.info(f"Initialized co-occurrence retriever with {len(documents)} documents")
    
    def _extract_concepts(self, document: Document) -> List[str]:
        """
        Extract key concepts from a document.
        
        Args:
            document: Document to extract concepts from
            
        Returns:
            List of concepts
        """
        content = document.page_content.lower()
        concepts = []
        
        # Travel-specific concept patterns
        travel_concepts = {
            # Allowances and rates
            "meal allowance", "accommodation", "incidental", "daily rate",
            "per diem", "kilometric rate", "mileage", "taxi", "rental car",
            
            # Travel types
            "temporary duty", "relocation", "deployment", "training",
            "conference", "meeting", "official travel", "personal travel",
            
            # Locations
            "canada", "united states", "international", "foreign",
            "isolated post", "special area", "headquarters",
            
            # Transportation
            "air travel", "ground transportation", "private vehicle",
            "government vehicle", "public transit", "pmv", "pomv",
            
            # Personnel
            "military member", "civilian employee", "dependant",
            "accompanying spouse", "contractor",
            
            # Administrative
            "claim", "advance", "authorization", "approval",
            "reimbursement", "receipt", "expense report"
        }
        
        # Check for presence of each concept
        for concept in travel_concepts:
            if concept in content:
                concepts.append(concept)
        
        # Extract specific values
        import re
        
        # Dollar amounts
        dollar_matches = re.findall(r'\$[\d,]+(?:\.\d{2})?', content)
        for match in dollar_matches[:3]:  # Limit to avoid too many
            concepts.append(f"amount:{match}")
        
        # Percentages
        percent_matches = re.findall(r'\d+(?:\.\d+)?%', content)
        for match in percent_matches[:2]:
            concepts.append(f"percentage:{match}")
        
        # Distances
        km_matches = re.findall(r'\d+\s*(?:km|kilometer|kilometre)', content)
        for match in km_matches[:2]:
            concepts.append(f"distance:{match}")
        
        return concepts
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """
        Get relevant documents using co-occurrence patterns.
        
        Args:
            query: Search query
            run_manager: Callback manager
            
        Returns:
            List of relevant documents
        """
        # Extract concepts from query
        query_doc = Document(page_content=query)
        query_concepts = self._extract_concepts(query_doc)
        
        logger.debug(f"Query concepts: {query_concepts}")
        
        # Expand concepts using co-occurrence
        expanded_concepts = set(query_concepts)
        for concept in query_concepts:
            related = self.index.get_related_concepts(concept, self.expansion_k)
            for related_concept, score in related:
                if score > 1:  # Only include if co-occurred more than once
                    expanded_concepts.add(related_concept)
        
        logger.debug(f"Expanded concepts: {expanded_concepts}")
        
        # Get documents containing these concepts
        doc_ids = self.index.get_documents_for_concepts(list(expanded_concepts))
        
        # Score documents by concept overlap
        doc_scores = []
        for doc_id in doc_ids:
            if doc_id in self.documents:
                doc = self.documents[doc_id]
                doc_concepts = self.index.document_map.get(doc_id, set())
                
                # Calculate score based on concept overlap
                overlap = len(doc_concepts.intersection(expanded_concepts))
                query_overlap = len(doc_concepts.intersection(query_concepts))
                
                # Boost documents that match original query concepts
                score = overlap + (query_overlap * 2)
                
                doc_scores.append((doc, score))
        
        # Sort by score and return top k
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        docs = [doc for doc, score in doc_scores[:self.k]]
        
        # Log retrieval
        self._log_event("retrieve", {
            "query": query,
            "query_concepts": query_concepts,
            "expanded_concepts": list(expanded_concepts),
            "num_candidates": len(doc_ids),
            "num_results": len(docs),
            "method": "cooccurrence"
        })
        
        return docs
    
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """
        Async get relevant documents.
        
        Co-occurrence retrieval is synchronous, so we wrap the sync call.
        """
        import asyncio
        return await asyncio.to_thread(
            self._get_relevant_documents,
            query,
            run_manager=run_manager
        )
    
    def update_index(self, new_documents: List[Document]):
        """
        Update the co-occurrence index with new documents.
        
        Args:
            new_documents: New documents to add to the index
        """
        for doc in new_documents:
            concepts = self._extract_concepts(doc)
            doc_id = doc.metadata.get("source", str(hash(doc.page_content)))
            
            # Add to index
            self.index.add_document(doc_id, concepts)
            
            # Add to document store
            self.documents[doc_id] = doc
        
        # Save updated index
        self.index.save()
        
        logger.info(f"Updated co-occurrence index with {len(new_documents)} new documents")
        
        self._log_event("update_index", {
            "num_new_documents": len(new_documents),
            "total_documents": len(self.documents)
        })