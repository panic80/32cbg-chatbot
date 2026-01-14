"""Prompt templates for different question types."""

from dataclasses import dataclass
from typing import Dict


@dataclass
class QuestionTypeConfig:
    """Configuration for a question type."""

    name: str
    description: str
    prompt_template: str
    examples: str


FACTUAL_PROMPT = """Based on the following document chunk, generate {n} factual questions that can be answered directly from the text.

Requirements:
- Questions should ask about specific facts, numbers, dates, rates, or definitions
- Answers must be directly extractable from the provided text
- Questions should be clear and unambiguous
- Include the specific answer from the text

Document chunk (ID: {chunk_id}):
---
{content}
---

Generate exactly {n} question-answer pairs in this JSON format:
```json
[
  {{
    "question": "What is the specific fact/rate/date mentioned?",
    "answer": "The exact answer from the text",
    "chunk_ids": ["{chunk_id}"]
  }}
]
```

Generate factual questions:"""


PROCEDURAL_PROMPT = """Based on the following document chunk, generate {n} procedural questions about how to do something or what steps are involved.

Requirements:
- Questions should ask about processes, procedures, or step-by-step instructions
- Questions should use phrases like "How do I...", "What are the steps to...", "What is the process for..."
- Answers should explain the procedure clearly
- Only ask about procedures that are actually described in the text

Document chunk (ID: {chunk_id}):
---
{content}
---

Generate exactly {n} question-answer pairs in this JSON format:
```json
[
  {{
    "question": "How do I [action]?",
    "answer": "Step-by-step explanation from the text",
    "chunk_ids": ["{chunk_id}"]
  }}
]
```

Generate procedural questions:"""


COMPARISON_PROMPT = """Based on the following document chunk, generate {n} comparison questions that compare different options, scenarios, or conditions.

Requirements:
- Questions should ask about differences, comparisons, or trade-offs
- Use phrases like "What is the difference between...", "How does X compare to Y...", "When should I use X vs Y..."
- Only create comparisons that are actually supported by the text
- If the text doesn't contain comparable items, focus on conditional questions (if/when scenarios)

Document chunk (ID: {chunk_id}):
---
{content}
---

Generate exactly {n} question-answer pairs in this JSON format:
```json
[
  {{
    "question": "What is the difference between X and Y?",
    "answer": "Explanation of the differences based on the text",
    "chunk_ids": ["{chunk_id}"]
  }}
]
```

Generate comparison questions:"""


QUESTION_TYPES: Dict[str, QuestionTypeConfig] = {
    "factual": QuestionTypeConfig(
        name="factual",
        description="Questions about specific facts, numbers, dates, or definitions",
        prompt_template=FACTUAL_PROMPT,
        examples="""
Examples of factual questions:
- "What is the meal rate for breakfast in Canada?"
- "What is the maximum kilometric rate for privately owned vehicles?"
- "What is the definition of temporary duty travel?"
""",
    ),
    "procedural": QuestionTypeConfig(
        name="procedural",
        description="Questions about processes, procedures, or step-by-step instructions",
        prompt_template=PROCEDURAL_PROMPT,
        examples="""
Examples of procedural questions:
- "How do I submit a travel claim?"
- "What are the steps to request a travel advance?"
- "What documentation is required for accommodation expenses?"
""",
    ),
    "comparison": QuestionTypeConfig(
        name="comparison",
        description="Questions comparing different options, scenarios, or conditions",
        prompt_template=COMPARISON_PROMPT,
        examples="""
Examples of comparison questions:
- "What is the difference between commercial and private accommodation rates?"
- "When should I use actual expenses vs allowances?"
- "How do day trip expenses differ from overnight travel?"
""",
    ),
}


def get_prompt(question_type: str, content: str, chunk_id: str, n: int = 2) -> str:
    """Get the prompt for generating questions of a specific type.

    Args:
        question_type: One of 'factual', 'procedural', 'comparison'
        content: The document chunk content
        chunk_id: The chunk ID for reference
        n: Number of questions to generate

    Returns:
        Formatted prompt string
    """
    if question_type not in QUESTION_TYPES:
        raise ValueError(f"Unknown question type: {question_type}")

    config = QUESTION_TYPES[question_type]
    return config.prompt_template.format(
        n=n,
        content=content,
        chunk_id=chunk_id,
    )


def get_all_types() -> list:
    """Get list of all question type names."""
    return list(QUESTION_TYPES.keys())


def get_type_description(question_type: str) -> str:
    """Get description for a question type."""
    if question_type not in QUESTION_TYPES:
        return ""
    return QUESTION_TYPES[question_type].description
