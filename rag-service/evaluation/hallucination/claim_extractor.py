"""Extract atomic claims from answers for hallucination detection."""

import asyncio
import json
import re
from typing import Any, List

from evaluation.core.results import Claim


# Patterns that indicate a claim is actually a citation/reference (should be filtered)
CITATION_PATTERNS = [
    r"^References?:",
    r"^Sources?:",
    r"^Citations?:",
    r"https?://",
    r"^According to .*(directive|document|section|policy)",
    r"^National Joint Council",
    r"^NJC Directive",
    r"section \d+\.\d+.*states",
    r"^\*\*References",
    r"^- National",
    r"retrieved from",
    r"available at:",
]


CLAIM_EXTRACTION_PROMPT = """Extract all factual claims from the following answer. Each claim should be:
- A single, verifiable statement
- Self-contained (understandable without context)
- Atomic (cannot be broken into smaller claims)
- Excludes opinions, hedging, or meta-statements about the answer itself
- Excludes source citations, references, URLs, or document identifiers (e.g., "According to Section X...", "Reference:", "https://...")

Answer to analyze:
---
{answer}
---

Extract claims as JSON array:
```json
[
  {{"claim": "The specific factual statement", "type": "fact|number|definition|procedure"}},
  ...
]
```

Extract claims:"""


class ClaimExtractor:
    """Extract atomic claims from answers using LLM."""

    def __init__(
        self,
        llm_client: Any = None,
        model: str = "gpt-4.1-mini",
        max_concurrent: int = 5,
    ):
        """Initialize claim extractor.

        Args:
            llm_client: Optional pre-configured LLM client
            model: Model to use if llm_client not provided
            max_concurrent: Maximum concurrent extractions
        """
        self.llm = llm_client
        self.model = model
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

        if self.llm is None:
            self._init_default_llm()

    def _init_default_llm(self):
        """Initialize default LLM client."""
        try:
            from langchain_openai import ChatOpenAI

            self.llm = ChatOpenAI(
                model=self.model,
                temperature=0.0,
            )
        except ImportError:
            print("Warning: langchain_openai not available")
            self.llm = None

    async def extract_claims(self, answer: str) -> List[Claim]:
        """Extract factual claims from answer text.

        Args:
            answer: The answer text to analyze

        Returns:
            List of extracted Claim objects
        """
        if not answer or not answer.strip():
            return []

        if self.llm is None:
            # Fallback: simple sentence splitting
            return self._simple_extraction(answer)

        async with self._semaphore:
            try:
                prompt = CLAIM_EXTRACTION_PROMPT.format(answer=answer)
                response = await self.llm.ainvoke(prompt)
                response_text = (
                    response.content if hasattr(response, "content") else str(response)
                )
                return self._parse_claims(response_text, answer)
            except Exception as e:
                print(f"Error extracting claims: {e}")
                return self._simple_extraction(answer)

    def _is_citation_claim(self, claim_text: str) -> bool:
        """Check if a claim is actually a citation/reference that should be filtered.

        Args:
            claim_text: The claim text to check

        Returns:
            True if this looks like a citation, False otherwise
        """
        for pattern in CITATION_PATTERNS:
            if re.search(pattern, claim_text, re.IGNORECASE):
                return True
        return False

    def _parse_claims(self, response: str, original_answer: str) -> List[Claim]:
        """Parse LLM response into Claim objects.

        Args:
            response: Raw LLM response
            original_answer: Original answer for context

        Returns:
            List of Claim objects
        """
        claims = []

        # Try to extract JSON
        json_match = re.search(r"\[[\s\S]*\]", response)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                for item in parsed:
                    claim_text = item.get("claim", "").strip()
                    # Filter out citation/reference claims
                    if claim_text and not self._is_citation_claim(claim_text):
                        claims.append(
                            Claim(
                                text=claim_text,
                                source_sentence=self._find_source_sentence(
                                    claim_text, original_answer
                                ),
                                claim_type=item.get("type", "fact"),
                            )
                        )
                return claims
            except json.JSONDecodeError:
                pass

        # Fallback: look for CLAIM: prefixed lines
        claim_pattern = re.compile(r"(?:CLAIM:|claim:|•|-)\s*(.+?)(?=(?:CLAIM:|claim:|•|-)|$)", re.DOTALL)
        for match in claim_pattern.finditer(response):
            claim_text = match.group(1).strip()
            # Filter out citation/reference claims
            if claim_text and len(claim_text) > 10 and not self._is_citation_claim(claim_text):
                claims.append(
                    Claim(
                        text=claim_text,
                        source_sentence=self._find_source_sentence(
                            claim_text, original_answer
                        ),
                        claim_type="fact",
                    )
                )

        return claims

    def _find_source_sentence(self, claim: str, answer: str) -> str:
        """Find the sentence in the answer that contains the claim.

        Args:
            claim: The extracted claim
            answer: Original answer text

        Returns:
            The source sentence or the claim itself
        """
        # Split answer into sentences
        sentences = re.split(r"[.!?]+", answer)

        # Find best matching sentence
        claim_words = set(claim.lower().split())
        best_match = claim
        best_overlap = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_words = set(sentence.lower().split())
            overlap = len(claim_words & sentence_words)

            if overlap > best_overlap:
                best_overlap = overlap
                best_match = sentence

        return best_match

    def _simple_extraction(self, answer: str) -> List[Claim]:
        """Simple fallback extraction without LLM.

        Args:
            answer: Answer text

        Returns:
            List of Claims (one per sentence)
        """
        claims = []

        # Split into sentences
        sentences = re.split(r"[.!?]+", answer)

        for sentence in sentences:
            sentence = sentence.strip()

            # Skip short sentences, questions, and meta-statements
            if len(sentence) < 20:
                continue
            if sentence.endswith("?"):
                continue
            # Skip citation/reference claims
            if self._is_citation_claim(sentence):
                continue
            if any(
                phrase in sentence.lower()
                for phrase in [
                    "i think",
                    "i believe",
                    "in my opinion",
                    "according to",
                    "it seems",
                    "might be",
                    "could be",
                ]
            ):
                continue

            # Determine claim type
            claim_type = "fact"
            if any(char.isdigit() for char in sentence):
                claim_type = "number"
            if any(
                phrase in sentence.lower()
                for phrase in ["defined as", "means", "refers to", "is a"]
            ):
                claim_type = "definition"
            if any(
                phrase in sentence.lower()
                for phrase in ["step", "first", "then", "next", "process", "procedure"]
            ):
                claim_type = "procedure"

            claims.append(
                Claim(
                    text=sentence,
                    source_sentence=sentence,
                    claim_type=claim_type,
                )
            )

        return claims

    async def extract_batch(self, answers: List[str]) -> List[List[Claim]]:
        """Extract claims from multiple answers.

        Args:
            answers: List of answer texts

        Returns:
            List of claim lists (one per answer)
        """
        tasks = [self.extract_claims(answer) for answer in answers]
        return await asyncio.gather(*tasks)
