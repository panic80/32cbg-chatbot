"""Query optimization and understanding pipeline."""

import re
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

from langchain_core.language_models import BaseLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger(__name__)


class QueryIntent(Enum):
    """Types of query intents."""
    RATE_LOOKUP = "rate_lookup"  # Looking for specific rates/values
    POLICY_QUESTION = "policy_question"  # Asking about policies/procedures
    ELIGIBILITY = "eligibility"  # Checking eligibility for benefits
    PROCESS_INQUIRY = "process_inquiry"  # How to do something
    DEFINITION = "definition"  # What is X?
    COMPARISON = "comparison"  # Comparing options
    CALCULATION = "calculation"  # Need calculation/computation
    GENERAL = "general"  # General inquiry
    RESTRICTION_LOOKUP = "restriction_lookup"  # Looking for limitations/restrictions
    AUTHORIZATION = "authorization"  # Checking what is authorized/permitted


class QueryClassification(BaseModel):
    """Query classification result."""
    intent: QueryIntent = Field(description="Primary intent of the query")
    entities: List[str] = Field(default_factory=list, description="Key entities mentioned")
    temporal_context: Optional[str] = Field(default=None, description="Time-related context")
    location_context: Optional[str] = Field(default=None, description="Location/jurisdiction context")
    requires_table_lookup: bool = Field(default=False, description="Whether query needs table data")
    is_class_a_context: bool = Field(
        default=False,
        description="Whether the query is explicitly about Class A Reserve service",
    )
    irregular_hours: bool = Field(
        default=False,
        description="Whether the scenario involves work outside normal hours",
    )
    ordered_outside_normal_hours: bool = Field(
        default=False,
        description="Whether the member was ordered/directed to work outside normal hours",
    )
    on_td_or_tasking: bool = Field(
        default=False,
        description="Whether the scenario places the member on TD, authorized tasking, or MTEC",
    )
    missed_meal_on_tasking: bool = Field(
        default=False,
        description="Whether the member missed a meal because of the tasking",
    )
    entitlement_likely_denied: bool = Field(
        default=False,
        description="Whether heuristics suggest entitlement should be denied",
    )
    confidence: float = Field(default=0.0, description="Classification confidence")


class QueryOptimizer:
    """Optimizes queries for better retrieval."""
    
    # Common abbreviations in travel context
    ABBREVIATIONS = {
        "TD": "travel directive",
        "POMV": "privately owned motor vehicle",
        "PMV": "private motor vehicle OR privately owned motor vehicle",
        "HG&E": "household goods and effects",
        "F&E": "furniture and effects",
        "CFRD": "Canadian Forces relocation directive",
        "IRP": "integrated relocation program",
        "CBI": "compensation and benefits instructions",
        "NJC": "national joint council",
        "POV": "privately owned vehicle",
        "km": "kilometer OR kilometres",
        "TBSE": "travel benefits and support element",
        "DCBA": "director compensation and benefits administration",
        "CAF": "Canadian Armed Forces",
        "CF": "Canadian Forces",
        "GMT": "Government Motor Transport OR Crown vehicle OR government vehicle",
        "mbr": "member",
        "approx": "approximately",
        "incl": "including",
        "excl": "excluding",
        "max": "maximum",
        "min": "minimum"
    }
    
    # Query templates for expansion
    EXPANSION_TEMPLATES = {
        QueryIntent.RATE_LOOKUP: [
            "{original} table rates allowance",
            "{original} per day daily amount",
            "{original} current rates Canada"
        ],
        QueryIntent.POLICY_QUESTION: [
            "{original} policy directive regulation",
            "{original} rules requirements conditions"
        ],
        QueryIntent.ELIGIBILITY: [
            "{original} eligible qualify entitlement",
            "{original} requirements conditions criteria"
        ],
        QueryIntent.PROCESS_INQUIRY: [
            "{original} how to process procedure",
            "{original} steps guide instructions"
        ]
    }
    
    def __init__(self, llm: Optional[BaseLLM] = None):
        """Initialize query optimizer."""
        self.llm = llm
        self._setup_classifier()
        
    def _setup_classifier(self):
        """Setup query classifier prompt."""
        if not self.llm:
            return
            
        self.parser = PydanticOutputParser(pydantic_object=QueryClassification)
        
        # Simplified prompt for faster classification (reduced from 8 considerations to concise format)
        self.classification_prompt = PromptTemplate(
            template="""Classify this Canadian Forces travel query concisely.

Query: {query}

Extract:
- intent: RATE_LOOKUP|ELIGIBILITY|PROCESS_INQUIRY|POLICY_QUESTION|COMPARISON|CALCULATION|DEFINITION|RESTRICTION_LOOKUP|AUTHORIZATION|GENERAL
- entities: key terms (rates, locations, benefits)
- requires_table_lookup: true if asking for specific values/rates
- is_class_a_context: true if Class A Reserve mentioned
- on_td_or_tasking: true if TD/tasking/MTEC mentioned
- irregular_hours: true if evening/night/weekend work mentioned
- ordered_outside_normal_hours: true if ordered to work outside normal hours
- missed_meal_on_tasking: true if missed meal due to tasking
- entitlement_likely_denied: true if Class A at home unit without TD/tasking

{format_instructions}""",
            input_variables=["query"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
    
    def expand_abbreviations(self, query: str) -> str:
        """Expand known abbreviations in the query."""
        expanded = query
        
        # Sort by length descending to avoid partial replacements
        sorted_abbrevs = sorted(
            self.ABBREVIATIONS.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
        
        for abbrev, full_form in sorted_abbrevs:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            
            # Check if abbreviation exists
            if re.search(pattern, expanded, re.IGNORECASE):
                # For GMT specifically, bias toward Crown vehicle phrasing the corpus uses
                if abbrev.upper() == "GMT":
                    replacement = f"{abbrev} (Crown vehicle / government vehicle / {full_form}) Crown vehicle government vehicle"
                else:
                    # Add full form in parentheses after abbreviation
                    replacement = f"{abbrev} ({full_form})"
                expanded = re.sub(pattern, replacement, expanded, flags=re.IGNORECASE)
            
        if expanded != query:
            logger.info(f"Expanded abbreviations: '{query}' -> '{expanded}'")

        return expanded

    def detect_and_expand_column_numbers(self, query: str) -> Tuple[str, Optional[str]]:
        """Detect column number queries and add exact match terms.

        Args:
            query: Original query string

        Returns:
            Tuple of (expanded_query, column_number_if_detected)
        """
        # Pattern to detect column number references
        # Matches: "column 17", "col 17", "column number 17", "column17"
        column_patterns = [
            r'\bcolumn\s*(?:number\s*)?(\d+)\b',
            r'\bcol\s*(?:number\s*)?(\d+)\b',
            r'\bcolumn(\d+)\b'
        ]

        for pattern in column_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                column_num = match.group(1)

                # Add minimal expansion - just the most specific term
                # The DOA PDF uses format "Column X –" (with en-dash)
                expanded = f"{query} Column {column_num}"

                logger.info(f"Detected column number query: column {column_num}")

                return (expanded, column_num)

        return (query, None)

    def _apply_classification_heuristics(
        self,
        classification: QueryClassification,
        query: str,
    ) -> QueryClassification:
        """Augment classification with rule-based heuristics."""
        query_lower = query.lower()
        entities = list(classification.entities or [])

        def add_entity(value: str) -> None:
            if not value:
                return
            if not any(existing.lower() == value.lower() for existing in entities):
                entities.append(value)

        # Location entities
        locations = [
            "canada",
            "usa",
            "united states",
            "ontario",
            "quebec",
            "alberta",
            "yukon",
            "nwt",
            "nunavut",
            "overseas",
            "international",
        ]
        if not classification.location_context:
            for loc in locations:
                if loc in query_lower:
                    classification.location_context = loc
                    add_entity(loc)
                    break
        else:
            add_entity(classification.location_context)

        # Benefit/rate types
        rate_types = [
            "meal",
            "breakfast",
            "lunch",
            "dinner",
            "incidental",
            "hotel",
            "accommodation",
            "kilometric",
            "mileage",
            "relocation",
            "posting",
        ]
        for rate in rate_types:
            if rate in query_lower:
                add_entity(rate)

        # Temporal context
        if not classification.temporal_context:
            if any(word in query_lower for word in ["2024", "2025", "current", "latest", "new"]):
                classification.temporal_context = "current"
            elif "retroactive" in query_lower or "previous" in query_lower:
                classification.temporal_context = "historical"

        # Table indicators
        if any(word in query_lower for word in ["table", "chart", "list", "schedule", "appendix"]):
            classification.requires_table_lookup = True

        # Detect Class A Reserve context
        class_a_terms = [
            "class a",
            "class-a",
            "primary reserve",
            "reservist",
            "reserve force",
            "parade night",
            "part-time service",
            "part time service",
            # Implicit strong indicators
            "parading",
            "drill night",
            "training night",
        ]
        if any(term in query_lower for term in class_a_terms):
            classification.is_class_a_context = True
            add_entity("class a")
            classification.confidence = max(classification.confidence or 0.0, 0.8)

        # Determine if the work occurs outside normal hours
        irregular_hours = classification.irregular_hours
        if not irregular_hours:
            time_24hr_matches = re.findall(r"\b([01]?\d|2[0-3])[0-5]\d\b", query_lower)
            for match in time_24hr_matches:
                hour = int(match[-2:]) if len(match) > 2 else int(match)
                if hour < 6 or hour >= 17:
                    irregular_hours = True
                    break

        if not irregular_hours:
            for hour_str, meridiem in re.findall(
                r"\b(\d{1,2})(?::\d{2})?\s*(a\.?m\.?|p\.?m\.?)\b",
                query_lower,
            ):
                hour = int(hour_str)
                meridiem_lower = meridiem.lower()
                if "p" in meridiem_lower and hour >= 5:
                    irregular_hours = True
                    break
                if "a" in meridiem_lower and hour < 6:
                    irregular_hours = True
                    break

        weekend_terms = [
            "weekend",
            "saturday",
            "sunday",
            "fri night",
            "friday night",
            "friday evening",
        ]
        mentions_weekend = any(term in query_lower for term in weekend_terms)

        if not irregular_hours:
            irregular_keywords = [
                "evening",
                "night",
                "after hours",
                "after-hours",
                "late",
                "weekend",
                "outside normal hours",
                "outside of normal hours",
            ]
            if any(keyword in query_lower for keyword in irregular_keywords):
                irregular_hours = True

        if mentions_weekend:
            irregular_hours = True

        classification.irregular_hours = classification.irregular_hours or irregular_hours

        if mentions_weekend:
            weekend_context_terms = [
                "training",
                "exercise",
                "course",
                "parade",
                "drill",
                "tasking",
                "duty",
                "muster",
                "overnight",
                "bivouac",
                "field",
            ]
            if classification.is_class_a_context or any(term in query_lower for term in weekend_context_terms):
                classification.on_td_or_tasking = True

        # Detect explicit orders to work outside normal hours
        ordered_indicators = [
            "ordered to",
            "ordered for",
            "ordered by",
            "tasked to",
            "tasked with",
            "directed to",
            "told to report",
            "required to report",
            "commanded to",
            "must report at",
            "mandatory formation",
            "mandatory training",
        ]
        if any(indicator in query_lower for indicator in ordered_indicators):
            classification.ordered_outside_normal_hours = True

        # Detect TD/tasking/MTEC indicators
        td_indicators = [
            " on td",
            " temporary duty",
            " temp duty",
            " on tasking",
            "tasking to",
            "tasking for",
            "tasked away",
            " mtec",
            " military temporary employment class",
            " away from home unit",
            " away from my unit",
            " away from unit",
            " attached posting",
            " detached duty",
            " out of town",
            " going away",
            " deployed",
        ]
        if any(indicator in query_lower for indicator in td_indicators):
            classification.on_td_or_tasking = True

        # Detect missed meal indicators
        meal_indicators = [
            "missed meal",
            "miss my meal",
            "missed lunch",
            "missed dinner",
            "missed breakfast",
            "no time to eat",
            "unable to eat",
            "skipped meal",
            "meal was denied",
            "didn't get a meal",
            "did not get a meal",
            "meal not provided",
            "miss a meal",
        ]
        if any(indicator in query_lower for indicator in meal_indicators):
            classification.missed_meal_on_tasking = True

        # Determine if entitlement is likely denied (heuristic)
        home_unit_indicators = [
            "home unit",
            "home armoury",
            "home armory",
            "unit training",
            "training night",
            "tuesday night training",
            "parade night",
            "parade",
            "local unit",
            "at the unit",
            "in my unit",
            "show up at",
        ]
        mentions_home_unit = any(indicator in query_lower for indicator in home_unit_indicators)
        classification.entitlement_likely_denied = bool(
            classification.is_class_a_context
            and not classification.on_td_or_tasking
            and not classification.ordered_outside_normal_hours
            and not classification.missed_meal_on_tasking
            and mentions_home_unit
        )

        classification.entities = entities
        return classification
        
    def simplify_complex_query(self, query: str) -> List[str]:
        """Break down complex queries into simpler sub-queries."""
        # Look for conjunctions that indicate multiple questions
        conjunctions = [" and ", " as well as ", " plus ", " also ", " furthermore "]
        
        # Check if query has multiple parts
        parts = [query]
        for conjunction in conjunctions:
            if conjunction in query.lower():
                # Split on conjunction
                split_parts = query.split(conjunction)
                if len(split_parts) > 1:
                    parts = []
                    for part in split_parts:
                        # Clean and add if substantial
                        cleaned = part.strip().rstrip(".,?")
                        if len(cleaned.split()) > 3:  # At least 4 words
                            parts.append(cleaned + "?")
                    break
                    
        # Also check for multiple question marks
        if parts == [query] and query.count("?") > 1:
            parts = [q.strip() + "?" for q in query.split("?") if q.strip()]
            
        if len(parts) > 1:
            logger.info(f"Simplified complex query into {len(parts)} parts")
            
        return parts
        
    async def classify_query(self, query: str) -> QueryClassification:
        """Classify the query intent and extract entities."""
        if not self.llm:
            # Fallback to rule-based classification
            return self._rule_based_classification(query)

        # Fast path: use rule-based for simple queries to avoid 7+ second LLM call
        # Simple query = short length AND clear intent keywords AND no complex entitlement context
        if self._is_simple_query(query):
            logger.info("Using fast-path rule-based classification for simple query")
            return self._rule_based_classification(query)

        try:
            # Use LLM for classification (complex queries only)
            prompt = self.classification_prompt.format(query=query)
            response = await self.llm.ainvoke(prompt)

            # Parse response
            classification = self.parser.parse(response.content)
            return self._apply_classification_heuristics(classification, query)

        except Exception as e:
            logger.warning(f"LLM classification failed: {e}, using rule-based")
            return self._rule_based_classification(query)

    def _is_simple_query(self, query: str) -> bool:
        """Check if query is simple enough for fast-path rule-based classification.

        Simple queries are:
        - Short (under 20 words)
        - Have clear intent keywords (rate, allowance, eligible, how to, etc.)
        - Don't involve complex entitlement scenarios (Class A, irregular hours, etc.)

        Returns True to use fast rule-based path, False to use LLM.
        """
        query_lower = query.lower()
        word_count = len(query.split())

        # Long queries need LLM for nuanced understanding
        if word_count > 25:
            return False

        # Complex entitlement scenarios need LLM analysis
        complex_indicators = [
            "class a", "class b", "class c",  # Reserve class context
            "home unit",  # Location context matters
            "evening", "night", "weekend", "after hours",  # Irregular hours
            "ordered", "directed", "tasking",  # Authorization context
            "missed meal", "entitled", "qualify",  # Entitlement determination
            "1700", "1800", "1900", "2000", "2100", "2200",  # Time indicators
        ]
        if any(indicator in query_lower for indicator in complex_indicators):
            return False

        # Simple queries with clear keywords can use fast path
        simple_keywords = [
            # Rate lookups
            "rate", "rates", "allowance", "$", "per diem", "amount", "cost",
            "appendix c", "appendix d", "table",
            # Simple definitions
            "what is", "what are", "define", "meaning of",
            # Simple process questions
            "how to", "how do i", "process", "procedure",
            # Policy questions without entitlement
            "policy", "directive", "rule",
        ]

        has_simple_keyword = any(kw in query_lower for kw in simple_keywords)
        is_short = word_count <= 15

        # Use fast path if: has clear keyword AND is short
        # OR: very short query (under 10 words) regardless of keywords
        return (has_simple_keyword and is_short) or word_count < 10
            
    def _rule_based_classification(self, query: str) -> QueryClassification:
        """Rule-based query classification fallback."""
        query_lower = query.lower()
        
        # Initialize classification
        classification = QueryClassification(
            intent=QueryIntent.GENERAL,
            entities=[],
            confidence=0.7
        )
        
        # Detect intent based on keywords
        if any(word in query_lower for word in ["rate", "amount", "cost", "price", "$", "dollar", "allowance"]):
            classification.intent = QueryIntent.RATE_LOOKUP
            classification.requires_table_lookup = True
            
        elif any(word in query_lower for word in ["eligible", "qualify", "entitled", "can i", "am i"]):
            classification.intent = QueryIntent.ELIGIBILITY
            
        elif any(word in query_lower for word in ["how to", "how do", "process", "procedure", "steps"]):
            classification.intent = QueryIntent.PROCESS_INQUIRY
            
        elif any(word in query_lower for word in ["policy", "directive", "regulation", "rule"]):
            classification.intent = QueryIntent.POLICY_QUESTION
            
        elif any(word in query_lower for word in ["what is", "what are", "define", "meaning"]):
            classification.intent = QueryIntent.DEFINITION
            
        elif any(word in query_lower for word in ["compare", "difference", "versus", "vs", "better"]):
            classification.intent = QueryIntent.COMPARISON
            
        elif any(word in query_lower for word in ["calculate", "computation", "total", "sum"]):
            classification.intent = QueryIntent.CALCULATION
            classification.requires_table_lookup = True

        classification = self._apply_classification_heuristics(classification, query)
            
        # Special handling for meal rate queries
        if ("meal" in query_lower and any(word in query_lower for word in ["rate", "allowance", "amount"])) or \
           ("appendix c" in query_lower) or ("appendix d" in query_lower):
            classification.intent = QueryIntent.RATE_LOOKUP
            classification.requires_table_lookup = True
            classification.confidence = 0.9
            
        return classification
        
    def expand_query(self, query: str, intent: QueryIntent) -> List[str]:
        """Expand query based on intent."""
        # Always include original, but allow domain-specific variant to lead
        expanded_queries: List[str] = []
        query_lower = query.lower()

        gmt_terms = ["gmt", "government motor transport", "crown vehicle", "government vehicle"]
        if any(term in query_lower for term in gmt_terms):
            primary_variant = f"{query} Crown vehicle government vehicle Government Motor Transport"
            expanded_queries.append(primary_variant)
        expanded_queries.append(query)
        
        # Get expansion templates for this intent
        templates = self.EXPANSION_TEMPLATES.get(intent, [])
        
        for template in templates:
            expanded = template.format(original=query)
            if expanded not in expanded_queries:
                expanded_queries.append(expanded)
                
        # Add specific expansions based on content
        # Meal-related expansions
        if "meal" in query_lower:
            if "yukon" not in query_lower:
                expanded_queries.append(f"{query} Yukon Alaska NWT Nunavut")
            expanded_queries.append(f"{query} breakfast lunch dinner rates")
            expanded_queries.append(f"{query} Appendix C Appendix D table")
            expanded_queries.append(f"{query} NJC meal allowance official rates")
            
        # Incidental expansions
        if "incidental" in query_lower:
            expanded_queries.append(f"{query} 17.30 13.00 daily allowance")
            
        # POMV/vehicle expansions
        if any(term in query_lower for term in ["pomv", "vehicle", "kilometric"]):
            pomv_expansion = f"{query} per kilometer cents privately owned"
            if pomv_expansion not in expanded_queries:
                expanded_queries.append(pomv_expansion)
        
        # GMT / Crown vehicle expansions
        if any(term in query_lower for term in ["gmt", "government motor transport", "crown vehicle", "government vehicle"]):
            gmt_variants = [
                query.replace("GMT", "Crown vehicle").replace("gmt", "Crown vehicle"),
                query.replace("GMT", "government vehicle").replace("gmt", "government vehicle"),
                f"{query} crown vehicle versus private motor vehicle government transport",
            ]
            # Prioritize the first variant by inserting it immediately after the original
            first_variant = gmt_variants.pop(0)
            if first_variant not in expanded_queries:
                expanded_queries.insert(1, first_variant)
            for variant in gmt_variants:
                if variant not in expanded_queries:
                    expanded_queries.append(variant)
            
        # Appendix C/D specific expansions
        if "appendix c" in query_lower or "appendix d" in query_lower:
            expanded_queries.append(f"{query} meal rates breakfast lunch dinner table")
            expanded_queries.append(f"{query} NJC official government rates")
            expanded_queries.append(f"{query} Canada USA international domestic")
            
        logger.info(f"Expanded query to {len(expanded_queries)} variants")
        return expanded_queries[:5]  # Limit to 5 expansions
        
    def detect_language(self, query: str) -> str:
        """Detect query language (basic implementation)."""
        # French indicators
        french_words = ["qu'est", "comment", "pourquoi", "quel", "quelle", 
                       "combien", "est-ce", "puis-je", "allocation", "taux"]
        
        query_lower = query.lower()
        french_count = sum(1 for word in french_words if word in query_lower)
        
        if french_count >= 2:
            return "fr"
        else:
            return "en"
            
    async def optimize_query(self, query: str, hyde_generator=None) -> Dict[str, Any]:
        """Full query optimization pipeline.

        Args:
            query: The user's query
            hyde_generator: Optional HyDEGenerator instance for hypothesis generation
        """
        # Detect language
        language = self.detect_language(query)

        # Expand abbreviations
        expanded = self.expand_abbreviations(query)

        # Detect and expand column numbers (for DOA queries)
        expanded, column_number = self.detect_and_expand_column_numbers(expanded)

        # Classify query
        classification = await self.classify_query(expanded)

        # Simplify if complex
        sub_queries = self.simplify_complex_query(expanded)

        # Expand based on intent
        all_expansions = []
        for sub_query in sub_queries:
            expansions = self.expand_query(sub_query, classification.intent)
            all_expansions.extend(expansions)

        # Remove duplicates while preserving order
        unique_expansions = []
        seen = set()
        for exp in all_expansions:
            if exp not in seen:
                seen.add(exp)
                unique_expansions.append(exp)

        # Generate HyDE hypothesis if enabled
        hyde_hypothesis = None
        if settings.enable_hyde and hyde_generator:
            hyde_hypothesis = await hyde_generator.generate_hypothesis(expanded)
            if hyde_hypothesis:
                logger.debug(f"Generated HyDE hypothesis: {hyde_hypothesis[:100]}...")

        return {
            "original_query": query,
            "language": language,
            "classification": classification.model_dump(),
            "expanded_queries": unique_expansions,
            "requires_translation": language != "en",
            "column_number": column_number,  # Add detected column number for filtering
            "hyde_hypothesis": hyde_hypothesis  # HyDE hypothetical answer for retrieval
        }
    
    def expand_query_for_retry(self, query: str) -> str:
        """Expand query with synonyms and related terms for better retrieval.
        
        Used in first retry iteration to cast a wider net.
        
        Args:
            query: Original query
            
        Returns:
            Expanded query with additional context terms
        """
        # Expand abbreviations
        expanded = self.expand_abbreviations(query)
        
        # Add domain context terms
        domain_terms = []
        query_lower = query.lower()
        
        # Add relevant context based on query content
        if any(term in query_lower for term in ["rate", "allowance", "cost", "amount"]):
            domain_terms.extend(["rates", "allowances", "amounts", "values"])
        if any(term in query_lower for term in ["meal", "food", "breakfast", "lunch", "dinner"]):
            domain_terms.extend(["meal allowance", "per diem", "daily rates"])
        if any(term in query_lower for term in ["travel", "trip", "journey"]):
            domain_terms.extend(["travel directive", "TD", "travel instructions"])
        if any(term in query_lower for term in ["vehicle", "car", "driving", "km"]):
            domain_terms.extend(["kilometric", "private vehicle", "POMV", "mileage"])
        if any(term in query_lower for term in ["gmt", "government motor transport", "crown vehicle", "government vehicle"]):
            domain_terms.extend(["Crown vehicle", "government vehicle", "Government Motor Transport"])
        if any(term in query_lower for term in ["hotel", "accommodation", "lodging"]):
            domain_terms.extend(["accommodation", "lodging", "hotel rates"])
            
        # Combine original with domain terms (avoid duplication)
        if domain_terms:
            expanded = f"{expanded} {' '.join(set(domain_terms))}"
            
        logger.debug(f"Expanded query: '{query}' -> '{expanded}'")
        return expanded
    
    def simplify_query_for_retry(self, query: str) -> str:
        """Simplify query to core terms, removing complex phrasing.
        
        Used in second retry iteration to focus on essential keywords.
        
        Args:
            query: Original query
            
        Returns:
            Simplified query with only core terms
        """
        # Remove question words and common phrases
        remove_patterns = [
            r'\b(what|where|when|why|how|can|could|would|should|do|does|did|is|are|am|was|were)\b',
            r'\b(the|a|an|my|your|our|their)\b',
            r'\b(please|kindly|help|tell|explain|show|give)\b',
            r'\b(me|you|us|them|i|we|they)\b',
        ]
        
        simplified = query.lower()
        for pattern in remove_patterns:
            simplified = re.sub(pattern, ' ', simplified, flags=re.IGNORECASE)
            
        # Clean up extra spaces
        simplified = ' '.join(simplified.split())
        
        # Expand abbreviations for clarity
        simplified = self.expand_abbreviations(simplified)
        
        # Extract key nouns and terms (keep numbers, proper nouns, domain terms)
        words = simplified.split()
        key_terms = []
        
        for word in words:
            # Keep if: has number, is long (>4 chars), or is known domain term
            if (
                any(c.isdigit() for c in word) or
                len(word) > 4 or
                word in self.ABBREVIATIONS or
                word in ["rate", "meal", "km", "trip", "cost", "travel", "caf", "td"]
            ):
                key_terms.append(word)
                
        simplified = ' '.join(key_terms) if key_terms else simplified
        
        logger.debug(f"Simplified query: '{query}' -> '{simplified}'")
        return simplified if simplified else query  # Fallback to original if empty


class QueryRewriter:
    """Rewrites queries for specific retrieval strategies."""
    
    def __init__(self, llm: Optional[BaseLLM] = None):
        """Initialize query rewriter."""
        self.llm = llm
        
    def rewrite_for_semantic_search(self, query: str, context: Dict[str, Any]) -> str:
        """Rewrite query optimized for semantic search."""
        # Add context terms for better semantic matching
        rewritten = query
        
        intent = context.get("classification", {}).get("intent")
        if intent == "rate_lookup":
            rewritten = f"specific rates values amounts {query}"
        elif intent == "policy_question":
            rewritten = f"official policy directive regulation {query}"
            
        return rewritten
        
    def rewrite_for_keyword_search(self, query: str, context: Dict[str, Any]) -> str:
        """Rewrite query optimized for keyword/BM25 search."""
        # Extract key terms and remove stop words
        stop_words = {"the", "a", "an", "is", "are", "what", "how", "can", "i", "my"}
        
        words = query.lower().split()
        keywords = [w for w in words if w not in stop_words]
        
        # Add synonyms for important terms
        synonym_map = {
            "rate": ["rate", "amount", "value"],
            "meal": ["meal", "breakfast", "lunch", "dinner", "food"],
            "travel": ["travel", "TD", "trip", "journey"],
            "claim": ["claim", "reimbursement", "expense"]
        }
        
        expanded_keywords = []
        for keyword in keywords:
            expanded_keywords.append(keyword)
            if keyword in synonym_map:
                expanded_keywords.extend(synonym_map[keyword])
                
        return " ".join(expanded_keywords)
        
    async def generate_hypothetical_answer(self, query: str) -> str:
        """Generate a hypothetical answer for HyDE retrieval."""
        if not self.llm:
            return query
            
        prompt = f"""Generate a hypothetical but realistic answer to this question about Canadian Forces travel policies. 
The answer should contain specific details that would be found in official documentation.

Question: {query}

Hypothetical Answer:"""
        
        try:
            response = await self.llm.ainvoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"Failed to generate hypothetical answer: {e}")
            return query
