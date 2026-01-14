"""Query enhancer for Class A Reservist context."""
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class ClassAQueryEnhancer:
    """Enhances queries with Class A Reserve context when appropriate."""
    
    # Keywords that indicate a query might be Class A related
    class_a_indicators = [
        "reserve", "reservist", "part-time", "part time",
        "weekend", "training", "parade", "course",
        "12 days", "35 days", "summer", "exercise",
        "TD", "temporary duty", "class a", "primary reserve"
    ]
    
    # Keywords that definitely indicate Class A context
    explicit_class_a = [
        "class a", "class-a", "class A", "CLASS A",
        "primary reserve", "part-time service"
    ]
    
    @classmethod
    def enhance_query(cls, query: str) -> str:
        """
        Enhance query with Class A context if appropriate.
        
        Args:
            query: Original user query
            
        Returns:
            Enhanced query with Class A context if applicable
        """
        query_lower = query.lower()
        
        # If query already explicitly mentions Class A, just return it
        if any(term in query_lower for term in cls.explicit_class_a):
            logger.debug(f"Query already contains explicit Class A reference: {query}")
            return query
        
        # Check if query might be Class A related
        if any(indicator in query_lower for indicator in cls.class_a_indicators):
            # Add Class A context
            enhanced = f"{query} (Class A Reserve context)"
            logger.info(f"Enhanced query with Class A context: {enhanced}")
            return enhanced
        
        # Check for specific scenarios that often apply to Class A
        class_a_scenarios = [
            ("meal allowance", "training"),
            ("travel", "course"),
            ("accommodation", "exercise"),
            ("mileage", "parade"),
            ("per diem", "weekend"),
            ("allowance", "reserve")
        ]
        
        for term1, term2 in class_a_scenarios:
            if term1 in query_lower or term2 in query_lower:
                enhanced = f"{query} (including Class A Reserve provisions)"
                logger.info(f"Enhanced query for Class A scenario: {enhanced}")
                return enhanced
        
        # No Class A context needed
        return query
    
    @classmethod
    def generate_class_a_variants(cls, query: str) -> List[str]:
        """
        Generate Class A specific query variants.
        
        Args:
            query: Original query
            
        Returns:
            List of query variants including Class A perspectives
        """
        variants = [query]  # Always include original
        
        # Add Class A specific variant
        if "class a" not in query.lower():
            variants.append(f"{query} for Class A Reservists")
            variants.append(f"{query} Primary Reserve part-time service")
        
        # Add training/exercise context if relevant
        if any(term in query.lower() for term in ["travel", "meal", "accommodation", "allowance"]):
            variants.append(f"{query} during Reserve training")
            variants.append(f"{query} weekend exercise Class A")
        
        # Add time-limited service context
        if any(term in query.lower() for term in ["per diem", "daily", "rate"]):
            variants.append(f"{query} Class A service under 12 days")
            variants.append(f"{query} Reserve training over 35 days")
        
        return list(set(variants))  # Remove duplicates
