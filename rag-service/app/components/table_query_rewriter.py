"""
Table Query Rewriter component for table-aware retrieval.

This module provides a query rewriter that optimizes queries
for retrieving table-based content in travel documents.
"""

from typing import List, Dict, Any, Optional
import logging
import re

from langchain_core.language_models import BaseLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from app.components.base import BaseComponent
from app.core.logging import get_logger
from app.utils.retry import with_retry_async
from app.utils.table_validator import TableValidator

logger = get_logger(__name__)


class RewrittenQuery(BaseModel):
    """Model for rewritten query output."""
    
    original_query: str = Field(description="Original user query")
    rewritten_query: str = Field(description="Rewritten query optimized for tables")
    table_keywords: List[str] = Field(description="Keywords to find in tables")
    value_patterns: List[str] = Field(description="Value patterns to search for")
    
    
TABLE_REWRITE_PROMPT = PromptTemplate(
    template="""You are a query rewriter specialized in Canadian Forces travel documents.
Your task is to rewrite queries to better find information in tables.

Original Query: {query}

Consider that travel documents often contain tables with:
- Meal allowances by location (breakfast, lunch, dinner rates)
- Kilometric rates by province (cents per km)
- Per diem rates by city
- Accommodation limits and hotel rates
- Travel time calculations
- Expense categories and limits

Common table headers include:
- Location, Province, City, Region
- Rate, Amount, Limit, Maximum
- Breakfast, Lunch, Dinner, Meals, Supper
- Kilometric, Per km, Mileage
- Daily, Per diem, Allowance
- Appendix C, Appendix D, NJC rates
- Canada, USA, International, Domestic

Rewrite the query to:
1. Include table-specific keywords (rate, table, schedule, appendix, limit)
2. Extract and preserve exact numeric values (e.g., "62.5 cents/km")
3. Add relevant location or category terms
4. Include variations of key terms (e.g., "kilometer" → "km", "kilometre")
5. Add context words that often appear near tables

IMPORTANT: If the query asks for a specific value (like "Ontario kilometric rate"), 
ensure the rewritten query includes both the location AND the type of rate.

{format_instructions}

Rewritten Query:""",
    input_variables=["query"],
    partial_variables={}
)


class TableQueryRewriter(BaseComponent):
    """
    Query rewriter optimized for table-based content retrieval.
    
    Enhances queries to better match table structures and content
    in travel documents.
    """
    
    def __init__(
        self,
        llm: Optional[BaseLLM] = None
    ):
        """
        Initialize the table query rewriter.
        
        Args:
            llm: Language model for query rewriting
        """
        super().__init__(component_type="rewriter", component_name="table_query")
        
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=RewrittenQuery)
        
        # Update prompt with format instructions
        self.prompt = TABLE_REWRITE_PROMPT.copy()
        self.prompt.partial_variables["format_instructions"] = self.parser.get_format_instructions()
        
        logger.info("Initialized table query rewriter")
    
    def rewrite_query(self, query: str) -> Dict[str, Any]:
        """
        Rewrite query for better table retrieval.
        
        Args:
            query: Original query
            
        Returns:
            Dictionary with rewritten query and metadata
        """
        # Quick rewrite for common patterns
        quick_rewrite = self._quick_rewrite(query)
        if quick_rewrite:
            return quick_rewrite
        
        # Use LLM for complex queries if available
        if self.llm:
            try:
                return self._llm_rewrite(query)
            except Exception as e:
                logger.warning(f"LLM rewrite failed: {e}, falling back to rule-based")
        
        # Fallback to rule-based rewrite
        return self._rule_based_rewrite(query)
    
    async def arewrite_query(self, query: str) -> Dict[str, Any]:
        """
        Async rewrite query for better table retrieval.
        
        Args:
            query: Original query
            
        Returns:
            Dictionary with rewritten query and metadata
        """
        # Quick rewrite for common patterns
        quick_rewrite = self._quick_rewrite(query)
        if quick_rewrite:
            return quick_rewrite
        
        # Use LLM for complex queries if available
        if self.llm:
            try:
                return await self._llm_rewrite_async(query)
            except Exception as e:
                logger.warning(f"LLM rewrite failed: {e}, falling back to rule-based")
        
        # Fallback to rule-based rewrite
        return self._rule_based_rewrite(query)
    
    def _quick_rewrite(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Quick pattern-based rewrite for common queries.
        
        Args:
            query: Original query
            
        Returns:
            Rewrite result or None
        """
        query_lower = query.lower()
        
        # Common table-related queries with enhanced patterns
        patterns = {
            r"meal.*(?:rate|allowance).*?(?:in|for)?\s*(\w+(?:\s+\w+)?)": {
                "keywords": ["meal", "rate", "table", "allowance", "breakfast", "lunch", "dinner", "food", "$", "dollar", "amount"],
                "template": "meal allowance rate table {location} breakfast lunch dinner dollar amounts"
            },
            r"(?:show|what|give).*meal.*(?:table|rates?)": {
                "keywords": ["meal", "table", "rate", "allowance", "breakfast", "lunch", "dinner", "$", "per meal"],
                "template": "meal allowance table rates breakfast lunch dinner dollar amounts"
            },
            r"(?:breakfast|lunch|dinner).*(?:rate|cost|amount).*?(?:in|for)?\s*(\w+(?:\s+\w+)?)": {
                "keywords": ["meal", "rate", "table", "allowance", "breakfast", "lunch", "dinner", "$", "amount"],
                "template": "{location} meal rates breakfast lunch dinner table dollar amounts"
            },
            r"(\w+(?:\s+\w+)?)\s*(?:kilometric|km|mileage).*rate": {
                "keywords": ["kilometric", "rate", "table", "km", "per", "kilometer", "cents", "mileage"],
                "template": "{location} kilometric rate table km cents per kilometer mileage"
            },
            r"(?:kilometric|km|mileage).*rate.*?(?:in|for)?\s*(\w+(?:\s+\w+)?)": {
                "keywords": ["kilometric", "rate", "table", "km", "per", "kilometer", "cents"],
                "template": "{location} kilometric rate table km cents per kilometer"
            },
            r"per diem.*?(?:in|for)?\s*(\w+(?:\s+\w+)?)": {
                "keywords": ["per diem", "daily", "rate", "allowance", "table", "expenses"],
                "template": "per diem daily rate table {location} allowance"
            },
            r"(?:hotel|accommodation).*(?:limit|rate).*?(?:in|for)?\s*(\w+(?:\s+\w+)?)": {
                "keywords": ["accommodation", "hotel", "limit", "rate", "table", "maximum", "lodging"],
                "template": "accommodation hotel limit rate table {location} maximum lodging"
            },
            r"what is the (\w+(?:\s+\w+)?)\s*(.*?)\s*rate": {
                "keywords": ["rate", "table", "schedule", "amount", "value"],
                "template": "{location} {rate_type} rate table schedule amount"
            },
            r"appendix\s*[cd]": {
                "keywords": ["appendix", "meal", "rate", "table", "breakfast", "lunch", "dinner", "NJC", "official"],
                "template": "Appendix C D meal rates table breakfast lunch dinner NJC official government"
            },
            r"meal.*(?:rate|allowance).*(?:appendix|table)": {
                "keywords": ["meal", "rate", "appendix", "table", "breakfast", "lunch", "dinner", "$", "NJC"],
                "template": "meal allowance rates Appendix C D table breakfast lunch dinner dollar amounts NJC"
            }
        }
        
        for pattern, config in patterns.items():
            match = re.search(pattern, query_lower)
            if match:
                # Extract captured groups
                template_vars = {}
                if match.lastindex:
                    if match.lastindex >= 1:
                        template_vars["location"] = match.group(1) if match.group(1) else ""
                    if match.lastindex >= 2:
                        template_vars["rate_type"] = match.group(2) if match.group(2) else ""
                
                # Format template with available variables
                try:
                    rewritten = config["template"].format(**template_vars)
                except KeyError:
                    # Fallback if template has missing variables
                    rewritten = config["template"].format(location="", rate_type="")
                
                # Clean up extra spaces
                rewritten = " ".join(rewritten.split())
                
                return {
                    "original_query": query,
                    "rewritten_query": rewritten,
                    "table_keywords": config["keywords"],
                    "value_patterns": self._extract_value_patterns(query),
                    "method": "quick_rewrite"
                }
        
        return None
    
    def _llm_rewrite(self, query: str) -> Dict[str, Any]:
        """
        Use LLM to rewrite query.
        
        Args:
            query: Original query
            
        Returns:
            Rewrite result
        """
        # Format prompt
        prompt_text = self.prompt.format(query=query)
        
        # Get LLM response
        response = self.llm.invoke(prompt_text)
        
        # Parse response
        try:
            result = self.parser.parse(response.content)
            return {
                "original_query": result.original_query,
                "rewritten_query": result.rewritten_query,
                "table_keywords": result.table_keywords,
                "value_patterns": result.value_patterns,
                "method": "llm_rewrite"
            }
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            raise
    
    @with_retry_async
    async def _llm_rewrite_async(self, query: str) -> Dict[str, Any]:
        """
        Async use LLM to rewrite query.
        
        Args:
            query: Original query
            
        Returns:
            Rewrite result
        """
        # Format prompt
        prompt_text = self.prompt.format(query=query)
        
        # Get LLM response
        response = await self.llm.ainvoke(prompt_text)
        
        # Parse response
        try:
            result = self.parser.parse(response.content)
            return {
                "original_query": result.original_query,
                "rewritten_query": result.rewritten_query,
                "table_keywords": result.table_keywords,
                "value_patterns": result.value_patterns,
                "method": "llm_rewrite"
            }
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            raise
    
    def _rule_based_rewrite(self, query: str) -> Dict[str, Any]:
        """
        Rule-based query rewriting.
        
        Args:
            query: Original query
            
        Returns:
            Rewrite result
        """
        query_lower = query.lower()
        keywords = []
        additions = []
        
        # Add table-related keywords
        if not any(word in query_lower for word in ["table", "schedule", "appendix", "rate"]):
            additions.append("table")
        
        # Detect query intent and add relevant keywords
        if any(word in query_lower for word in ["meal", "breakfast", "lunch", "dinner"]):
            keywords.extend(["meal", "allowance", "rate", "food", "$", "dollar", "amount", "per meal"])
            additions.extend(["meal rate table", "dollar amounts", "breakfast lunch dinner"])
            
        if any(word in query_lower for word in ["hotel", "accommodation", "lodging"]):
            keywords.extend(["accommodation", "hotel", "limit", "rate"])
            additions.append("accommodation limit table")
            
        if any(word in query_lower for word in ["km", "kilometer", "mileage", "driving"]):
            keywords.extend(["kilometric", "rate", "km", "mileage"])
            additions.append("kilometric rate table")
            
        if "per diem" in query_lower or "daily" in query_lower:
            keywords.extend(["per diem", "daily", "rate", "allowance"])
            additions.append("per diem rate table")
        
        # Special handling for trip planning queries
        is_trip_planning = any(word in query_lower for word in ["trip", "travel", "journey", "planning"])
        needs_cost_info = any(word in query_lower for word in ["cost", "expense", "estimate", "budget", "how much"])
        
        if is_trip_planning and needs_cost_info:
            # Add all rate-related keywords for comprehensive retrieval
            keywords.extend(["meal", "allowance", "rate", "kilometric", "incidental", "table", "$"])
            additions.extend([
                "meal allowances breakfast lunch dinner rates",
                "kilometric rates per km",
                "incidental allowances daily rates",
                "travel expense rates table"
            ])
        
        # Extract locations
        locations = self._extract_locations(query)
        if locations:
            keywords.extend(locations)
        
        # Build rewritten query
        rewritten_parts = [query]
        if additions:
            rewritten_parts.extend(additions)
        
        rewritten_query = " ".join(rewritten_parts)
        
        # Extract value patterns
        value_patterns = self._extract_value_patterns(query)
        
        return {
            "original_query": query,
            "rewritten_query": rewritten_query,
            "table_keywords": list(set(keywords)),
            "value_patterns": value_patterns,
            "method": "rule_based"
        }
    
    def _extract_locations(self, query: str) -> List[str]:
        """
        Extract location names from query.
        
        Args:
            query: Query text
            
        Returns:
            List of locations
        """
        locations = []
        query_lower = query.lower()
        
        # Canadian provinces and territories
        provinces = [
            "ontario", "quebec", "british columbia", "alberta", "manitoba",
            "saskatchewan", "nova scotia", "new brunswick", "newfoundland",
            "prince edward island", "northwest territories", "yukon", "nunavut"
        ]
        
        # Major cities
        cities = [
            "ottawa", "toronto", "montreal", "vancouver", "calgary", "edmonton",
            "winnipeg", "halifax", "quebec city", "victoria", "regina", "saskatoon",
            "st. john's", "charlottetown", "yellowknife", "whitehorse", "iqaluit"
        ]
        
        # Check for provinces
        for province in provinces:
            if province in query_lower:
                locations.append(province)
        
        # Check for cities
        for city in cities:
            if city in query_lower:
                locations.append(city)
        
        # Check for country names
        if "canada" in query_lower:
            locations.append("canada")
        if any(us in query_lower for us in ["united states", "usa", "u.s."]):
            locations.append("united states")
        
        return locations
    
    def _extract_value_patterns(self, query: str) -> List[str]:
        """
        Extract value patterns from query with enhanced numeric detection.
        
        Args:
            query: Query text
            
        Returns:
            List of value patterns
        """
        patterns = []
        
        # Dollar amounts with various formats
        dollar_patterns = [
            r'\$[\d,]+(?:\.\d{2})?',  # $123.45 or $1,234
            r'[\d,]+(?:\.\d{2})?\s*(?:dollars?|cents?)',  # 123.45 dollars
            r'[\d,]+(?:\.\d{2})?\s*¢',  # 62.5¢
        ]
        for pattern in dollar_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            patterns.extend(matches)
        
        # Percentages
        percent_matches = re.findall(r'\d+(?:\.\d+)?%', query)
        patterns.extend(percent_matches)
        
        # Distances with various units
        distance_patterns = [
            r'\d+(?:\.\d+)?\s*(?:km|kilometer|kilometre)',
            r'\d+(?:\.\d+)?\s*(?:miles?|mi)',
            r'\d+(?:\.\d+)?\s*cents?/km',  # rates like "62.5 cents/km"
        ]
        for pattern in distance_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            patterns.extend(matches)
        
        # Time periods
        time_matches = re.findall(r'\d+\s*(?:hours?|days?|weeks?|months?)', query, re.IGNORECASE)
        patterns.extend(time_matches)
        
        # Numeric ranges
        range_matches = re.findall(r'\d+\s*(?:to|-)\s*\d+', query)
        patterns.extend(range_matches)
        
        # Extract any numeric values that might be in tables
        for word in query.split():
            numeric_val = TableValidator.extract_numeric_value(word)
            if numeric_val is not None:
                patterns.append(word)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_patterns = []
        for pattern in patterns:
            if pattern not in seen:
                seen.add(pattern)
                unique_patterns.append(pattern)
        
        return unique_patterns