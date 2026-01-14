"""Query enhancement strategies for the unified retrieval framework."""

from typing import List, Dict, Any, Optional
import asyncio
from langchain_core.language_models import BaseLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser

from app.unified_retrieval.strategies.base import BaseStrategy, RetrievalContext, StrategyType
from app.core.logging import get_logger
from app.services.cache import cache_result
from app.services.llm_pool import get_llm_pool

logger = get_logger(__name__)


class LineListOutputParser(BaseOutputParser[List[str]]):
    """Parse output as a list of lines."""
    
    def parse(self, text: str) -> List[str]:
        """Parse output text into a list of queries."""
        lines = text.strip().split("\n")
        queries = []
        for line in lines:
            cleaned = line.strip()
            # Remove list markers
            cleaned = cleaned.lstrip("0123456789.-*) ")
            if cleaned:
                queries.append(cleaned)
        return queries
    
    @property
    def _type(self) -> str:
        return "line_list"


class MultiQueryStrategy(BaseStrategy):
    """
    Generate multiple query variations using LLM.
    
    This strategy creates different versions of the original query to improve
    retrieval coverage by capturing different phrasings and perspectives.
    """
    
    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        num_queries: int = 5,
        include_original: bool = True,
        prompt_template: Optional[PromptTemplate] = None,
        **kwargs
    ):
        """
        Initialize the multi-query strategy.
        
        Args:
            llm: Language model to use for query generation
            num_queries: Number of query variations to generate
            include_original: Whether to include the original query
            prompt_template: Custom prompt template
        """
        super().__init__(
            strategy_type=StrategyType.QUERY_ENHANCEMENT,
            name="multi_query_strategy",
            **kwargs
        )
        self.llm = llm  # LLM should be provided by the pipeline
        self.num_queries = num_queries
        self.include_original = include_original
        self.parser = LineListOutputParser()
        
        self.prompt_template = prompt_template or PromptTemplate(
            input_variables=["question", "num_queries"],
            template="""You are an AI assistant helping to search Canadian Forces travel instructions.
Generate {num_queries} different versions of the given user question to retrieve relevant 
documents from a vector database. Focus on travel-specific terminology and values.

Original question: {question}

Generate different versions that:
1. Focus on specific values, rates, or amounts mentioned
2. Use alternative travel terminology (e.g., per diem, allowance, reimbursement)
3. Include relevant context (domestic, international, military)
4. Break down compound questions into focused parts
5. Consider both general policies and specific rates
6. Include Class A Reserve perspective when applicable
7. For vehicle/driving queries, include variants for restrictions and authorization
8. For distance queries, include kilometer/km limitations

Generate one query per line:"""
        )
    
    @cache_result(ttl=3600, key_prefix="multi_query")
    async def _generate_queries(self, original_query: str) -> List[str]:
        """Generate query variations using LLM."""
        try:
            # Generate queries directly with LLM
            prompt = self.prompt_template.format(
                question=original_query,
                num_queries=self.num_queries
            )
            
            # Call LLM directly
            result = await self.llm.ainvoke(prompt)
            
            # Parse the result
            queries = self.parser.parse(result.content if hasattr(result, 'content') else str(result))
            
            # Include original if requested
            if self.include_original:
                queries = [original_query] + queries
            
            # Deduplicate while preserving order
            seen = set()
            unique_queries = []
            for q in queries:
                if q.lower() not in seen:
                    seen.add(q.lower())
                    unique_queries.append(q)
            
            self._log_event(
                "queries_generated",
                {
                    "original_query": original_query,
                    "num_generated": len(unique_queries),
                    "queries": unique_queries[:3]  # Log first 3 for debugging
                }
            )
            
            return unique_queries
            
        except Exception as e:
            logger.error(f"Error generating queries: {e}")
            # Fallback to original query
            return [original_query]
    
    async def execute(self, context: RetrievalContext) -> RetrievalContext:
        """Execute the multi-query enhancement."""
        original_query = context.get_query()
        
        # Generate query variations
        queries = await self._generate_queries(original_query)
        
        # Store queries in metadata
        context.metadata["generated_queries"] = queries
        context.metadata["query_count"] = len(queries)
        
        # Add strategy output
        context.add_strategy_output(
            self.component_name,
            {
                "queries": queries,
                "original_query": original_query
            }
        )
        
        return context


class SelfQueryStrategy(BaseStrategy):
    """
    Convert natural language to metadata filters.
    
    This strategy analyzes the query to extract structured filters
    for metadata-based retrieval.
    """
    
    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        metadata_field_info: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        """
        Initialize the self-query strategy.
        
        Args:
            llm: Language model to use
            metadata_field_info: Information about available metadata fields
        """
        super().__init__(
            strategy_type=StrategyType.QUERY_ENHANCEMENT,
            name="self_query_strategy",
            **kwargs
        )
        self.llm = llm  # LLM should be provided by the pipeline
        self.metadata_field_info = metadata_field_info or self._get_default_metadata_info()
    
    def _get_default_metadata_info(self) -> List[Dict[str, Any]]:
        """Get default metadata field information."""
        return [
            {
                "name": "source",
                "description": "Source document name",
                "type": "string"
            },
            {
                "name": "chapter",
                "description": "Chapter in the document",
                "type": "string"
            },
            {
                "name": "section",
                "description": "Section in the document",
                "type": "string"
            },
            {
                "name": "subsection",
                "description": "Subsection in the document",
                "type": "string"
            },
            {
                "name": "doc_type",
                "description": "Type of document (policy, guide, form)",
                "type": "string"
            },
            {
                "name": "applies_to",
                "description": "Who the content applies to",
                "type": "string"
            },
            {
                "name": "effective_date",
                "description": "When the content became effective",
                "type": "string"
            }
        ]
    
    async def _extract_filters(self, query: str) -> Dict[str, Any]:
        """Extract metadata filters from query."""
        prompt = PromptTemplate(
            input_variables=["query", "metadata_fields"],
            template="""Analyze the following query and extract any metadata filters that could help find relevant documents.

Query: {query}

Available metadata fields:
{metadata_fields}

Extract filters in JSON format. Only include filters that are clearly mentioned or strongly implied in the query.
For example:
- "chapter 5" -> {{"chapter": "5"}}
- "Class A reservist travel" -> {{"applies_to": "Class A"}}
- "international travel rates" -> {{"doc_type": "rates", "travel_type": "international"}}

Return only the JSON object with filters, or an empty object {{}} if no filters are found:"""
        )
        
        try:
            # Format metadata fields
            fields_str = "\n".join([
                f"- {field['name']}: {field['description']} ({field['type']})"
                for field in self.metadata_field_info
            ])
            
            # Get LLM response
            response = await self.llm.ainvoke(
                prompt.format(query=query, metadata_fields=fields_str)
            )
            
            # Parse JSON response
            import json
            filters = json.loads(response.content.strip())
            
            self._log_event(
                "filters_extracted",
                {
                    "query": query,
                    "filters": filters
                }
            )
            
            return filters
            
        except Exception as e:
            logger.error(f"Error extracting filters: {e}")
            return {}
    
    async def execute(self, context: RetrievalContext) -> RetrievalContext:
        """Execute the self-query enhancement."""
        query = context.get_query()
        
        # Extract metadata filters
        filters = await self._extract_filters(query)
        
        # Update context filters
        context.filters.update(filters)
        
        # Add strategy output
        context.add_strategy_output(
            self.component_name,
            {
                "extracted_filters": filters,
                "query": query
            }
        )
        
        return context


class QueryExpansionStrategy(BaseStrategy):
    """
    Expand queries with synonyms and domain terms.
    
    This strategy enriches the query with related terms, synonyms,
    and domain-specific vocabulary to improve recall.
    """
    
    def __init__(
        self,
        expansion_terms: Optional[Dict[str, List[str]]] = None,
        llm: Optional[BaseLLM] = None,
        use_llm: bool = True,
        **kwargs
    ):
        """
        Initialize the query expansion strategy.
        
        Args:
            expansion_terms: Dictionary of terms and their expansions
            llm: Language model for dynamic expansion
            use_llm: Whether to use LLM for expansion
        """
        super().__init__(
            strategy_type=StrategyType.QUERY_ENHANCEMENT,
            name="query_expansion_strategy",
            **kwargs
        )
        self.expansion_terms = expansion_terms or self._get_default_expansions()
        self.llm = llm  # LLM should be provided by the pipeline if use_llm else None
        self.use_llm = use_llm
    
    def _get_default_expansions(self) -> Dict[str, List[str]]:
        """Get default expansion terms for CAF travel domain."""
        return {
            "travel": ["trip", "journey", "transportation", "movement"],
            "allowance": ["payment", "reimbursement", "compensation", "entitlement"],
            "per diem": ["daily rate", "daily allowance", "subsistence"],
            "meal": ["food", "sustenance", "rations", "dining"],
            "accommodation": ["lodging", "hotel", "quarters", "shelter"],
            "class a": ["reservist", "part-time", "reserve force"],
            "claim": ["request", "submission", "application"],
            "rate": ["amount", "value", "cost", "price"],
            "kilometer": ["km", "distance", "mileage"],
            "vehicle": ["car", "automobile", "transportation", "pmv"]
        }
    
    async def _expand_with_llm(self, query: str) -> str:
        """Use LLM to expand query with synonyms."""
        prompt = PromptTemplate(
            input_variables=["query"],
            template="""Expand the following travel-related query with relevant synonyms and related terms.
Focus on Canadian Forces travel terminology.

Original query: {query}

Provide an expanded version that includes key synonyms and related terms while maintaining the original meaning.
Keep it concise and relevant:"""
        )
        
        try:
            response = await self.llm.ainvoke(prompt.format(query=query))
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error in LLM expansion: {e}")
            return query
    
    def _expand_with_dictionary(self, query: str) -> str:
        """Expand query using predefined dictionary."""
        expanded_parts = []
        words = query.lower().split()
        
        for word in words:
            # Check if word has expansions
            if word in self.expansion_terms:
                # Add original and expansions
                expanded_parts.append(f"({word} OR {' OR '.join(self.expansion_terms[word])})")
            else:
                expanded_parts.append(word)
        
        return " ".join(expanded_parts)
    
    async def execute(self, context: RetrievalContext) -> RetrievalContext:
        """Execute the query expansion."""
        query = context.get_query()
        
        # Expand query
        if self.use_llm and self.llm:
            expanded_query = await self._expand_with_llm(query)
        else:
            expanded_query = self._expand_with_dictionary(query)
        
        # Update context
        context.enhanced_query = expanded_query
        context.metadata["original_query_before_expansion"] = query
        
        # Add strategy output
        context.add_strategy_output(
            self.component_name,
            {
                "original_query": query,
                "expanded_query": expanded_query,
                "method": "llm" if self.use_llm else "dictionary"
            }
        )
        
        return context


class AbbreviationExpansionStrategy(BaseStrategy):
    """Handle CAF-specific abbreviations using a curated dictionary."""
    
    def __init__(
        self,
        abbreviations: Optional[Dict[str, str]] = None,
        include_both: bool = True,
        **kwargs
    ):
        """
        Initialize the abbreviation expansion strategy.
        
        Args:
            abbreviations: Optional custom abbreviation mapping to extend the defaults
            include_both: Include both abbreviation and expansion in the rewritten query
        """
        super().__init__(
            strategy_type=StrategyType.QUERY_ENHANCEMENT,
            name="abbreviation_expansion_strategy",
            **kwargs
        )
        self.include_both = include_both
        self.abbreviations = (abbreviations or self._get_legacy_abbreviations())
    
    def _get_legacy_abbreviations(self) -> Dict[str, str]:
        """Get legacy CAF abbreviations for backward compatibility."""
        return {
            "caf": "canadian armed forces",
            "cf": "canadian forces",
            "ta": "travel authority",
            "td": "temporary duty",
            "pmv": "personal motor vehicle",
            "pomv": "privately owned motor vehicle",
            "cbi": "compensation and benefits instructions",
            "qr&o": "queens regulations and orders",
            "dnd": "department of national defence",
            "hq": "headquarters",
            "co": "commanding officer",
            "ncm": "non-commissioned member",
            "res f": "reserve force",
            "reg f": "regular force",
            "lwop": "leave without pay",
            "ir": "imposed restriction",
            "pld": "post living differential",
            "sea": "separation expense allowance",
            "rsa": "relocation service account",
            "hg&e": "household goods and effects",
            "dhh": "dependent household and effects",
            "km": "kilometer",
            "govt": "government",
            "gmt": "government motor transport"
        }
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand abbreviations in text."""
        # Expand using the configured abbreviation dictionary
        words = text.split()
        expanded_words = []
        
        for word in words:
            lower_word = word.lower().strip(".,!?;:()[]{}\"'")
            if lower_word in self.abbreviations:
                expansion = self.abbreviations[lower_word]
                if self.include_both:
                    expanded_words.append(f"({word} OR {expansion})")
                else:
                    expanded_words.append(expansion)
            else:
                expanded_words.append(word)
        
        return " ".join(expanded_words)
    
    async def execute(self, context: RetrievalContext) -> RetrievalContext:
        """Execute the abbreviation expansion."""
        query = context.get_query()
        
        # Expand abbreviations
        expanded_query = self._expand_abbreviations(query)
        
        # Check if any expansions were made
        if expanded_query != query:
            context.enhanced_query = expanded_query
            context.metadata["abbreviations_expanded"] = True
        else:
            context.metadata["abbreviations_expanded"] = False
        
        # Add strategy output
        context.add_strategy_output(
            self.component_name,
            {
                "original_query": query,
                "expanded_query": expanded_query if expanded_query != query else None,
                "expansions_made": expanded_query != query
            }
        )
        
        return context
