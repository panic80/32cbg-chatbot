"""Advanced Query Expansion for Canadian Forces Travel Domain

This module implements sophisticated query expansion techniques including
domain-specific synonyms, contextual expansion, entity extraction, and
temporal context handling.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
import spacy
from langchain_core.messages import BaseMessage
import logging


logger = logging.getLogger(__name__)


class AdvancedQueryExpander:
    """Advanced query expansion with domain-specific enhancements"""
    
    def __init__(self, spacy_model: str = "en_core_web_sm", use_centralized_glossary: bool = True):
        """Initialize query expander
        
        Args:
            spacy_model: SpaCy model for NLP tasks
            use_centralized_glossary: Deprecated flag retained for compatibility; centralized glossary has been removed
        """
        try:
            self.nlp = spacy.load(spacy_model)
        except:
            logger.warning(f"SpaCy model {spacy_model} not found, using basic expansion")
            self.nlp = None
        
        self.use_centralized_glossary = use_centralized_glossary
        if use_centralized_glossary:
            logger.warning("Centralized glossary support has been removed; using built-in abbreviations")
        # Use legacy abbreviations
        self.abbreviations = self._get_legacy_abbreviations()
        
        # Canadian Forces domain-specific synonyms
        self.domain_synonyms = {
            # Travel terms
            "travel": ["trip", "journey", "movement", "deployment"],
            "claim": ["reimbursement", "expense", "submission", "request"],
            "per diem": ["daily allowance", "meal allowance", "daily rate"],
            "advance": ["prepayment", "travel advance", "upfront payment"],
            "authorization": ["approval", "permission", "clearance"],
            
            # Military terms
            "CAF": ["Canadian Armed Forces", "Canadian Forces", "military"],
            "member": ["soldier", "personnel", "service member", "military member"],
            "posting": ["assignment", "transfer", "relocation", "move"],
            "TD": ["temporary duty", "temp duty", "TDY"],
            "leave": ["vacation", "time off", "absence", "furlough"],
            
            # Document terms
            "form": ["document", "paperwork", "application"],
            "policy": ["regulation", "directive", "guideline", "rule"],
            "procedure": ["process", "method", "steps", "instructions"],
            
            # Financial terms
            "allowance": ["benefit", "entitlement", "payment"],
            "rate": ["amount", "level", "percentage"],
            "expense": ["cost", "expenditure", "charge"],
            "receipt": ["proof", "invoice", "bill", "documentation"],
            
            # Location terms
            "domestic": ["within Canada", "national", "internal"],
            "international": ["foreign", "overseas", "abroad", "outside Canada"],
            "base": ["garrison", "station", "installation", "camp"]
        }
        
        # Temporal expressions
        self.temporal_patterns = {
            "recent": lambda: f"after {(datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')}",
            "current": lambda: f"as of {datetime.now().strftime('%Y-%m-%d')}",
            "latest": lambda: f"most recent version",
            "new": lambda: f"updated after {(datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')}",
            "upcoming": lambda: f"effective after {datetime.now().strftime('%Y-%m-%d')}",
            "previous": lambda: f"before {datetime.now().strftime('%Y-%m-%d')}",
            "last year": lambda: f"in {datetime.now().year - 1}",
            "this year": lambda: f"in {datetime.now().year}"
        }
        
        # Entity type mappings
        self.entity_mappings = {
            "location": ["base", "city", "province", "country", "region"],
            "rank": ["private", "corporal", "sergeant", "officer", "general"],
            "unit": ["regiment", "battalion", "squadron", "division", "brigade"],
            "document": ["form", "policy", "directive", "order", "manual"]
        }
    
    def expand_query(
        self,
        query: str,
        conversation_history: Optional[List[BaseMessage]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Expand query with multiple enhancement techniques
        
        Args:
            query: Original query
            conversation_history: Previous conversation messages
            context: Additional context
            
        Returns:
            Dictionary with expanded query and metadata
        """
        expanded_terms = set()
        entities = {}
        temporal_context = []
        
        # 1. Expand abbreviations
        expanded_query = self._expand_abbreviations(query)
        
        # 2. Extract entities
        if self.nlp:
            entities = self._extract_entities(expanded_query)
        
        # 3. Domain-specific synonym expansion
        synonyms = self._expand_synonyms(expanded_query)
        expanded_terms.update(synonyms)
        
        # 4. Contextual expansion from conversation history
        if conversation_history:
            contextual_terms = self._expand_from_context(
                expanded_query, 
                conversation_history
            )
            expanded_terms.update(contextual_terms)
        
        # 5. Temporal context expansion
        temporal_context = self._expand_temporal(expanded_query)
        
        # 6. Entity-based expansion
        entity_terms = self._expand_entities(entities)
        expanded_terms.update(entity_terms)
        
        # Build final expanded query
        final_query = self._build_expanded_query(
            expanded_query,
            expanded_terms,
            temporal_context
        )
        
        return {
            "original_query": query,
            "expanded_query": final_query,
            "expanded_terms": list(expanded_terms),
            "entities": entities,
            "temporal_context": temporal_context,
            "abbreviations_expanded": expanded_query != query
        }
    
    def _expand_abbreviations(self, query: str) -> str:
        """Expand known abbreviations in query"""
        # Expand using the built-in abbreviation mappings
        expanded = query
        
        for abbr, full in self.abbreviations.items():
            # Match whole words only
            pattern = r'\b' + re.escape(abbr) + r'\b'
            expanded = re.sub(
                pattern,
                f"{abbr} ({full})",
                expanded,
                flags=re.IGNORECASE
            )
        
        return expanded
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract named entities from query"""
        if not self.nlp:
            return {}
        
        doc = self.nlp(query)
        entities = {}
        
        # Standard NER
        for ent in doc.ents:
            entity_type = ent.label_.lower()
            if entity_type not in entities:
                entities[entity_type] = []
            entities[entity_type].append(ent.text)
        
        # Custom pattern matching for military entities
        # Rank patterns
        rank_pattern = r'\b(private|corporal|sergeant|warrant officer|lieutenant|captain|major|colonel|general)\b'
        ranks = re.findall(rank_pattern, query, re.IGNORECASE)
        if ranks:
            entities["rank"] = ranks
        
        # Unit patterns
        unit_pattern = r'\b(\d+\s*(regiment|battalion|squadron|division|brigade))\b'
        units = re.findall(unit_pattern, query, re.IGNORECASE)
        if units:
            entities["unit"] = [unit[0] for unit in units]
        
        return entities
    
    def _expand_synonyms(self, query: str) -> Set[str]:
        """Expand domain-specific synonyms"""
        expanded_terms = set()
        query_lower = query.lower()
        
        for term, synonyms in self.domain_synonyms.items():
            if term in query_lower:
                expanded_terms.update(synonyms)
        
        # Also check if any synonym appears and add the main term
        for term, synonyms in self.domain_synonyms.items():
            for synonym in synonyms:
                if synonym in query_lower:
                    expanded_terms.add(term)
                    expanded_terms.update(synonyms)
        
        return expanded_terms
    
    def _expand_from_context(
        self,
        query: str,
        conversation_history: List[BaseMessage]
    ) -> Set[str]:
        """Expand query based on conversation context"""
        contextual_terms = set()
        
        # Extract key terms from recent conversation
        recent_messages = conversation_history[-5:]  # Last 5 messages
        
        for message in recent_messages:
            content = message.content.lower()
            
            # Look for mentioned entities
            for entity_type, patterns in self.entity_mappings.items():
                for pattern in patterns:
                    if pattern in content:
                        contextual_terms.add(pattern)
            
            # Look for domain terms
            for term in self.domain_synonyms.keys():
                if term in content:
                    contextual_terms.add(term)
        
        return contextual_terms
    
    def _expand_temporal(self, query: str) -> List[str]:
        """Expand temporal expressions"""
        temporal_context = []
        query_lower = query.lower()
        
        for pattern, expander in self.temporal_patterns.items():
            if pattern in query_lower:
                temporal_context.append(expander())
        
        # Date pattern matching
        date_patterns = [
            (r'\b(\d{4})\b', lambda m: f"year:{m.group(1)}"),
            (r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})\b',
             lambda m: f"month:{m.group(1)} year:{m.group(2)}"),
            (r'\b(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})\b',
             lambda m: f"date:{m.group(1)} month:{m.group(2)} year:{m.group(3)}")
        ]
        
        for pattern, formatter in date_patterns:
            matches = re.finditer(pattern, query_lower, re.IGNORECASE)
            for match in matches:
                temporal_context.append(formatter(match))
        
        return temporal_context
    
    def _expand_entities(self, entities: Dict[str, List[str]]) -> Set[str]:
        """Expand based on extracted entities"""
        entity_terms = set()
        
        # Add entity type keywords
        for entity_type, values in entities.items():
            if entity_type in self.entity_mappings:
                entity_terms.update(self.entity_mappings[entity_type])
            
            # Add the entity values themselves
            entity_terms.update(values)
        
        return entity_terms
    
    def _build_expanded_query(
        self,
        base_query: str,
        expanded_terms: Set[str],
        temporal_context: List[str]
    ) -> str:
        """Build final expanded query"""
        parts = [base_query]
        
        # Add expanded terms
        if expanded_terms:
            # Filter out terms already in base query
            new_terms = [
                term for term in expanded_terms 
                if term.lower() not in base_query.lower()
            ]
            if new_terms:
                parts.append(f"({' OR '.join(new_terms)})")
        
        # Add temporal context
        if temporal_context:
            parts.extend(temporal_context)
        
        return " ".join(parts)
    
    def get_query_type(self, query: str) -> str:
        """Determine query type for optimization
        
        Returns:
            Query type: 'policy', 'procedure', 'rate', 'form', 'general'
        """
        query_lower = query.lower()
        
        # Policy queries
        if any(term in query_lower for term in ["policy", "regulation", "directive", "rule"]):
            return "policy"
        
        # Procedure queries
        if any(term in query_lower for term in ["how to", "procedure", "process", "steps"]):
            return "procedure"
        
        # Rate/financial queries
        if any(term in query_lower for term in ["rate", "amount", "cost", "allowance", "per diem"]):
            return "rate"
        
        # Form/document queries
        if any(term in query_lower for term in ["form", "document", "application", "template"]):
            return "form"
        
        return "general"
    
    def _get_legacy_abbreviations(self) -> Dict[str, str]:
        """Get legacy abbreviations for backward compatibility."""
        return {
            "td": "temporary duty",
            "tdy": "temporary duty",
            "caf": "canadian armed forces",
            "cf": "canadian forces",
            "dnd": "department of national defence",
            "lta": "leave travel assistance",
            "irp": "integrated relocation program",
            "pmv": "personal motor vehicle",
            "pov": "privately owned vehicle",
            "gmt": "government motor transport",
            "pld": "post living differential",
            "foa": "financial operations and administration",
            "cfao": "canadian forces administrative order",
            "qr&o": "queen's regulations and orders",
            "daod": "defence administrative orders and directives",
            "hg&e": "household goods and effects",
            "f&e": "furniture and effects",
            "ir": "imposed restriction",
            "se": "separation expense",
            "outcan": "outside canada"
        }
