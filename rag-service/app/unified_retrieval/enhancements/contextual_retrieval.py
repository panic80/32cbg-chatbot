"""Contextual Retrieval with Conversation Memory

This module implements a conversation memory system that tracks entities and topics
to enhance retrieval relevance based on conversation context.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from collections import deque, defaultdict
import numpy as np
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from langchain_classic.memory import ConversationSummaryMemory
from langchain_core.embeddings import Embeddings
import logging

logger = logging.getLogger(__name__)


class ConversationMemory:
    """Tracks conversation context including entities, topics, and relevance"""
    
    def __init__(self, max_history: int = 10):
        """Initialize conversation memory
        
        Args:
            max_history: Maximum number of turns to remember
        """
        self.max_history = max_history
        self.message_history = deque(maxlen=max_history)
        self.entities: Dict[str, Set[str]] = defaultdict(set)
        self.topics: List[str] = []
        self.topic_scores: Dict[str, float] = defaultdict(float)
        self.turn_count = 0
    
    def add_turn(
        self,
        query: str,
        response: str,
        entities: Dict[str, List[str]],
        topics: List[str]
    ):
        """Add a conversation turn to memory"""
        self.turn_count += 1
        
        # Store message
        self.message_history.append({
            "turn": self.turn_count,
            "query": query,
            "response": response,
            "entities": entities,
            "topics": topics,
            "timestamp": datetime.now()
        })
        
        # Update entities
        for entity_type, values in entities.items():
            self.entities[entity_type].update(values)
        
        # Update topics with decay
        for topic in topics:
            self.topic_scores[topic] = self.topic_scores.get(topic, 0) + 1.0
        
        # Decay older topics
        for topic in list(self.topic_scores.keys()):
            if topic not in topics:
                self.topic_scores[topic] *= 0.9
                if self.topic_scores[topic] < 0.1:
                    del self.topic_scores[topic]
    
    def get_active_entities(self) -> Dict[str, List[str]]:
        """Get currently active entities"""
        return {k: list(v) for k, v in self.entities.items() if v}
    
    def get_active_topics(self) -> List[Tuple[str, float]]:
        """Get active topics sorted by relevance"""
        return sorted(
            self.topic_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of conversation context"""
        recent_queries = [
            turn["query"] 
            for turn in list(self.message_history)[-3:]
        ]
        
        return {
            "turn_count": self.turn_count,
            "active_entities": self.get_active_entities(),
            "active_topics": self.get_active_topics()[:5],
            "recent_queries": recent_queries
        }


class ContextualRetriever:
    """Retriever that uses conversation context to enhance relevance"""
    
    def __init__(
        self,
        embeddings: Embeddings,
        memory_size: int = 10,
        entity_boost: float = 0.2,
        topic_boost: float = 0.15,
        continuity_boost: float = 0.1
    ):
        """Initialize contextual retriever
        
        Args:
            embeddings: Embeddings model
            memory_size: Size of conversation memory
            entity_boost: Score boost for entity matches
            topic_boost: Score boost for topic relevance
            continuity_boost: Score boost for topic continuity
        """
        self.embeddings = embeddings
        self.memory = ConversationMemory(max_history=memory_size)
        self.entity_boost = entity_boost
        self.topic_boost = topic_boost
        self.continuity_boost = continuity_boost
        
        # Topic extraction patterns
        self.topic_patterns = {
            "travel_claim": ["claim", "reimbursement", "expense", "submission"],
            "per_diem": ["per diem", "meal", "allowance", "daily rate"],
            "authorization": ["approval", "authorization", "permission"],
            "posting": ["posting", "relocation", "transfer", "move"],
            "leave": ["leave", "vacation", "LTA", "time off"],
            "policy": ["policy", "regulation", "directive", "rule"],
            "international": ["international", "foreign", "overseas", "abroad"],
            "accommodation": ["hotel", "lodging", "accommodation", "quarters"]
        }
    
    def update_context(
        self,
        query: str,
        response: str,
        documents: List[Document]
    ):
        """Update conversation context with new turn"""
        # Extract entities from query and response
        entities = self._extract_entities(query + " " + response)
        
        # Extract topics
        topics = self._extract_topics(query, documents)
        
        # Update memory
        self.memory.add_turn(query, response, entities, topics)
    
    def enhance_retrieval(
        self,
        query: str,
        documents: List[Document],
        scores: Optional[List[float]] = None
    ) -> Tuple[List[Document], List[float]]:
        """Enhance document retrieval with contextual scoring
        
        Args:
            query: Current query
            documents: Retrieved documents
            scores: Optional base scores
            
        Returns:
            Tuple of (reranked documents, adjusted scores)
        """
        if not documents:
            return documents, []
        
        # Get conversation context
        context = self.memory.get_context_summary()
        
        # Initialize scores if not provided
        if scores is None:
            scores = [1.0] * len(documents)
        
        # Calculate contextual scores
        enhanced_scores = []
        for i, (doc, base_score) in enumerate(zip(documents, scores)):
            # Entity-based scoring
            entity_score = self._calculate_entity_score(doc, context["active_entities"])
            
            # Topic relevance scoring
            topic_score = self._calculate_topic_score(doc, context["active_topics"])
            
            # Topic continuity scoring
            continuity_score = self._calculate_continuity_score(
                query, 
                doc,
                context["recent_queries"]
            )
            
            # Combine scores
            final_score = base_score * (
                1.0 + 
                self.entity_boost * entity_score +
                self.topic_boost * topic_score +
                self.continuity_boost * continuity_score
            )
            
            enhanced_scores.append(final_score)
        
        # Rerank documents
        sorted_indices = np.argsort(enhanced_scores)[::-1]
        reranked_docs = [documents[i] for i in sorted_indices]
        reranked_scores = [enhanced_scores[i] for i in sorted_indices]
        
        # Add context metadata
        for doc, score in zip(reranked_docs, reranked_scores):
            doc.metadata["contextual_score"] = score
            doc.metadata["context_turn"] = context["turn_count"]
        
        return reranked_docs, reranked_scores
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text"""
        entities = defaultdict(list)
        text_lower = text.lower()
        
        # Simple pattern-based extraction
        # Locations
        location_patterns = [
            r'\b(ottawa|toronto|halifax|edmonton|winnipeg|calgary|vancouver)\b',
            r'\b(ontario|quebec|alberta|british columbia|manitoba|nova scotia)\b'
        ]
        for pattern in location_patterns:
            import re
            matches = re.findall(pattern, text_lower)
            if matches:
                entities["location"].extend(matches)
        
        # Military units
        unit_pattern = r'\b(\d+\s*(regiment|battalion|squadron|wing|division))\b'
        unit_matches = re.findall(unit_pattern, text_lower)
        if unit_matches:
            entities["unit"].extend([m[0] for m in unit_matches])
        
        # Forms/documents
        form_pattern = r'\b(cf\s*\d+|dnd\s*\d+|form\s*\d+)\b'
        form_matches = re.findall(form_pattern, text_lower)
        if form_matches:
            entities["form"].extend(form_matches)
        
        return dict(entities)
    
    def _extract_topics(self, query: str, documents: List[Document]) -> List[str]:
        """Extract topics from query and documents"""
        topics = []
        
        # Check query against topic patterns
        query_lower = query.lower()
        for topic, keywords in self.topic_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                topics.append(topic)
        
        # Extract topics from document content
        for doc in documents[:3]:  # Check top 3 documents
            content_lower = doc.page_content.lower()
            for topic, keywords in self.topic_patterns.items():
                if topic not in topics:
                    keyword_count = sum(1 for kw in keywords if kw in content_lower)
                    if keyword_count >= 2:  # At least 2 keywords
                        topics.append(topic)
        
        return topics
    
    def _calculate_entity_score(
        self,
        document: Document,
        active_entities: Dict[str, List[str]]
    ) -> float:
        """Calculate entity-based relevance score"""
        if not active_entities:
            return 0.0
        
        score = 0.0
        doc_content = document.page_content.lower()
        doc_metadata = str(document.metadata).lower()
        
        for entity_type, values in active_entities.items():
            for value in values:
                if value.lower() in doc_content or value.lower() in doc_metadata:
                    score += 1.0
                    # Bonus for metadata matches
                    if value.lower() in doc_metadata:
                        score += 0.5
        
        # Normalize by total entities
        total_entities = sum(len(v) for v in active_entities.values())
        return min(score / max(total_entities, 1), 1.0)
    
    def _calculate_topic_score(
        self,
        document: Document,
        active_topics: List[Tuple[str, float]]
    ) -> float:
        """Calculate topic relevance score"""
        if not active_topics:
            return 0.0
        
        score = 0.0
        doc_content = document.page_content.lower()
        
        for topic, weight in active_topics:
            if topic in self.topic_patterns:
                keywords = self.topic_patterns[topic]
                matches = sum(1 for kw in keywords if kw in doc_content)
                if matches > 0:
                    score += weight * (matches / len(keywords))
        
        # Normalize
        total_weight = sum(w for _, w in active_topics)
        return min(score / max(total_weight, 1), 1.0)
    
    def _calculate_continuity_score(
        self,
        current_query: str,
        document: Document,
        recent_queries: List[str]
    ) -> float:
        """Calculate topic continuity score"""
        if not recent_queries:
            return 0.0
        
        # Check if document relates to recent query topics
        doc_content = document.page_content.lower()
        continuity_score = 0.0
        
        for recent_query in recent_queries:
            # Simple keyword overlap
            query_words = set(recent_query.lower().split())
            doc_words = set(doc_content.split())
            
            overlap = len(query_words & doc_words)
            if overlap > 0:
                continuity_score += overlap / len(query_words)
        
        return min(continuity_score / len(recent_queries), 1.0)
    
    def get_context_enhanced_query(self, query: str) -> str:
        """Enhance query with conversation context"""
        context = self.memory.get_context_summary()
        
        # Add active entities to query
        entity_terms = []
        for entity_type, values in context["active_entities"].items():
            if values and len(values) <= 3:  # Limit to avoid query explosion
                entity_terms.extend(values)
        
        # Add top topics
        topic_terms = []
        for topic, score in context["active_topics"][:3]:
            if topic in self.topic_patterns and score > 0.5:
                topic_terms.extend(self.topic_patterns[topic][:2])
        
        # Build enhanced query
        parts = [query]
        if entity_terms:
            parts.append(f"({' OR '.join(entity_terms)})")
        if topic_terms:
            parts.append(f"({' OR '.join(topic_terms)})")
        
        return " ".join(parts)
