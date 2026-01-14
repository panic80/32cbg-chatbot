"""Metadata extraction service for automatic document enrichment."""

import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import dateutil.parser
from collections import Counter

from langchain_core.documents import Document
from langchain_core.language_models import BaseLLM

from app.core.logging import get_logger
from app.models.documents import DocumentType

logger = get_logger(__name__)


class MetadataExtractor:
    """Extract metadata from document content automatically."""
    
    def __init__(self, llm: Optional[BaseLLM] = None):
        """Initialize metadata extractor.
        
        Args:
            llm: Optional LLM for advanced extraction
        """
        self.llm = llm
        
        # Regex patterns for common metadata
        self.patterns = {
            "policy_number": [
                r'(?:Policy|Directive|Regulation)\s*(?:No\.?|Number|#)?\s*([A-Z0-9\-\.]+)',
                r'(?:TB|DND|CF)\s*[A-Z]{2,}\s*\d{3,}',
                r'Chapter\s*(\d+(?:\.\d+)*)'
            ],
            "date": [
                r'(?:Effective|Updated|Modified|Published|Date)[:\s]+([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
                r'(\d{4}-\d{2}-\d{2})',
                r'(\d{1,2}/\d{1,2}/\d{4})',
                r'(\d{1,2}\s+[A-Za-z]+\s+\d{4})'
            ],
            "version": [
                r'(?:Version|Rev|Revision)[\s:]+([0-9\.]+)',
                r'v\.?\s*([0-9\.]+)',
                r'(?:Draft|Final)\s*(?:Version)?\s*([0-9\.]+)?'
            ],
            "organization": [
                r'(?:Department of|Ministry of)\s+([A-Z][A-Za-z\s]+)',
                r'(?:DND|CAF|RCAF|RCN|CA)\b',
                r'(?:Treasury Board|National Defence|Canadian Forces)'
            ],
            "classification": [
                r'(?:UNCLASSIFIED|PROTECTED|SECRET|TOP SECRET)',
                r'(?:PUBLIC|INTERNAL USE|CONFIDENTIAL)'
            ]
        }
        
        # Keywords for categorization
        self.category_keywords = {
            "travel": ["travel", "trip", "journey", "transportation", "accommodation", "per diem", "expenses"],
            "policy": ["policy", "directive", "regulation", "guideline", "standard", "procedure"],
            "benefits": ["benefit", "entitlement", "allowance", "compensation", "reimbursement"],
            "leave": ["leave", "vacation", "absence", "time off", "holiday"],
            "pay": ["pay", "salary", "wage", "remuneration", "earnings"],
            "training": ["training", "course", "education", "development", "certification"],
            "health": ["health", "medical", "dental", "wellness", "safety"],
            "pension": ["pension", "retirement", "superannuation", "annuity"]
        }
        
    async def extract_metadata(self, document: Document) -> Dict[str, Any]:
        """Extract metadata from document content.
        
        Args:
            document: Document to extract metadata from
            
        Returns:
            Dictionary of extracted metadata
        """
        content = document.page_content
        existing_metadata = document.metadata.copy()
        
        # Extract using patterns
        pattern_metadata = self._extract_pattern_metadata(content)
        
        # Extract categories and topics
        categories = self._extract_categories(content)
        topics = self._extract_topics(content)
        
        # Extract entities
        entities = self._extract_entities(content)
        
        # Extract summary if LLM available
        if self.llm:
            summary = await self._extract_llm_summary(content)
        else:
            summary = self._extract_basic_summary(content)
            
        # Combine all metadata
        extracted_metadata = {
            **pattern_metadata,
            "categories": categories,
            "topics": topics,
            "entities": entities,
            "auto_summary": summary,
            "extraction_timestamp": datetime.utcnow().isoformat(),
            "has_tables": self._detect_tables(content),
            "has_lists": self._detect_lists(content),
            "language": self._detect_language(content),
            "readability_score": self._calculate_readability(content)
        }
        
        # Merge with existing metadata (existing takes precedence)
        final_metadata = {**extracted_metadata, **existing_metadata}
        
        return final_metadata
        
    def _extract_pattern_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata using regex patterns."""
        metadata = {}
        
        # Extract policy numbers
        for pattern in self.patterns["policy_number"]:
            match = re.search(pattern, content[:1000], re.IGNORECASE)
            if match:
                metadata["policy_number"] = match.group(1)
                break
                
        # Extract dates
        dates_found = []
        for pattern in self.patterns["date"]:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                try:
                    # Try to parse the date
                    date_str = match if isinstance(match, str) else match[0]
                    parsed_date = dateutil.parser.parse(date_str)
                    dates_found.append(parsed_date)
                except:
                    continue
                    
        if dates_found:
            # Use the most recent date as last_modified
            metadata["detected_date"] = max(dates_found).isoformat()
            
        # Extract version
        for pattern in self.patterns["version"]:
            match = re.search(pattern, content[:500], re.IGNORECASE)
            if match:
                metadata["version"] = match.group(1) if match.group(1) else "1.0"
                break
                
        # Extract organization
        orgs_found = []
        for pattern in self.patterns["organization"]:
            matches = re.findall(pattern, content[:1000], re.IGNORECASE)
            orgs_found.extend(matches)
            
        if orgs_found:
            # Get most common organization
            org_counter = Counter(orgs_found)
            metadata["organization"] = org_counter.most_common(1)[0][0]
            
        # Extract classification
        for pattern in self.patterns["classification"]:
            match = re.search(pattern, content[:200], re.IGNORECASE)
            if match:
                metadata["classification"] = match.group(0).upper()
                break
                
        return metadata
        
    def _extract_categories(self, content: str) -> List[str]:
        """Extract document categories based on keywords."""
        content_lower = content.lower()
        categories = []
        
        for category, keywords in self.category_keywords.items():
            # Count keyword occurrences
            keyword_count = sum(1 for keyword in keywords if keyword in content_lower)
            
            # Add category if multiple keywords found
            if keyword_count >= 2:
                categories.append(category)
                
        return categories[:3]  # Return top 3 categories
        
    def _extract_topics(self, content: str) -> List[str]:
        """Extract main topics from content."""
        # Simple topic extraction based on capitalized phrases
        topics = []
        
        # Find section headers (lines that are mostly capitalized)
        lines = content.split('\n')
        for line in lines[:50]:  # Check first 50 lines
            line = line.strip()
            if len(line) > 10 and len(line) < 100:
                # Check if line is mostly uppercase or title case
                if line.isupper() or (line[0].isupper() and ' ' in line):
                    # Clean and add as topic
                    topic = line.title() if line.isupper() else line
                    topic = re.sub(r'^\d+\.?\s*', '', topic)  # Remove numbering
                    if topic and len(topic) > 5:
                        topics.append(topic)
                        
        # Deduplicate and return top topics
        seen = set()
        unique_topics = []
        for topic in topics:
            if topic not in seen:
                seen.add(topic)
                unique_topics.append(topic)
                
        return unique_topics[:5]
        
    def _extract_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract named entities from content."""
        entities = {
            "locations": [],
            "organizations": [],
            "monetary": [],
            "dates": []
        }
        
        # Extract locations (cities, provinces, countries)
        location_patterns = [
            r'\b(?:Ottawa|Toronto|Vancouver|Halifax|Edmonton|Victoria|Quebec City)\b',
            r'\b(?:Ontario|Quebec|British Columbia|Alberta|Manitoba|Saskatchewan|Nova Scotia)\b',
            r'\b(?:Canada|United States|US|USA|UK|United Kingdom)\b'
        ]
        for pattern in location_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            entities["locations"].extend(matches)
            
        # Extract monetary amounts
        money_pattern = r'\$[\d,]+(?:\.\d{2})?|\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|CAD|USD)\b'
        money_matches = re.findall(money_pattern, content, re.IGNORECASE)
        entities["monetary"] = list(set(money_matches))[:10]
        
        # Extract organizations (beyond basic patterns)
        org_pattern = r'\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,3}(?:\s+(?:Inc|Corp|Ltd|Limited|Department|Ministry|Agency|Board|Commission))?\b'
        org_matches = re.findall(org_pattern, content)
        entities["organizations"] = list(set(org_matches))[:10]
        
        # Dates are already extracted in pattern metadata
        
        # Deduplicate entities
        for key in entities:
            entities[key] = list(set(entities[key]))
            
        return entities
        
    async def _extract_llm_summary(self, content: str) -> str:
        """Extract summary using LLM."""
        if not self.llm:
            return self._extract_basic_summary(content)
            
        # Limit content for LLM processing
        content_preview = content[:2000]
        
        prompt = f"""Provide a brief 2-3 sentence summary of this document:

{content_preview}

Summary:"""

        try:
            response = await self.llm.ainvoke(prompt)
            summary = response.content if hasattr(response, 'content') else str(response)
            return summary.strip()
        except Exception as e:
            logger.warning(f"LLM summary extraction failed: {e}")
            return self._extract_basic_summary(content)
            
    def _extract_basic_summary(self, content: str) -> str:
        """Extract basic summary without LLM."""
        # Get first paragraph or first 200 characters
        paragraphs = content.split('\n\n')
        for para in paragraphs:
            para = para.strip()
            if len(para) > 50:  # Meaningful paragraph
                return para[:200] + "..." if len(para) > 200 else para
                
        # Fallback to first 200 characters
        return content[:200] + "..." if len(content) > 200 else content
        
    def _detect_tables(self, content: str) -> bool:
        """Detect if content contains tables."""
        # Check for markdown tables
        if '|' in content and content.count('|') > 4:
            return True
        # Check for HTML tables
        if '<table' in content.lower():
            return True
        # Check for tab-separated values
        if '\t' in content and content.count('\t') > 4:
            return True
        return False
        
    def _detect_lists(self, content: str) -> bool:
        """Detect if content contains lists."""
        list_patterns = [
            r'^\s*[-*•]\s+',  # Bullet points
            r'^\s*\d+\.\s+',   # Numbered lists
            r'^\s*[a-z]\.\s+', # Lettered lists
        ]
        
        for pattern in list_patterns:
            if re.search(pattern, content, re.MULTILINE | re.IGNORECASE):
                return True
        return False
        
    def _detect_language(self, content: str) -> str:
        """Detect document language."""
        # Simple detection based on common words
        french_words = ["le", "la", "de", "et", "un", "une", "des", "les", "pour", "dans"]
        english_words = ["the", "and", "of", "to", "in", "for", "is", "with", "that", "by"]
        
        content_lower = content.lower()
        words = content_lower.split()[:100]  # Check first 100 words
        
        french_count = sum(1 for word in words if word in french_words)
        english_count = sum(1 for word in words if word in english_words)
        
        if french_count > english_count:
            return "fr"
        else:
            return "en"
            
    def _calculate_readability(self, content: str) -> float:
        """Calculate simple readability score (0-100)."""
        # Simple readability based on average sentence and word length
        sentences = re.split(r'[.!?]+', content)
        words = content.split()
        
        if not sentences or not words:
            return 50.0
            
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Simple formula (lower is more readable)
        score = 100 - min(100, (avg_sentence_length * 2 + avg_word_length * 10))
        
        return round(max(0, score), 1)