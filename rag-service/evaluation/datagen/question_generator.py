"""LLM-based Q&A pair generation from document chunks."""

import asyncio
import json
import random
import re
from typing import Any, Dict, List, Optional

from evaluation.datagen.dataset import EvaluationDataset, GeneratedQA
from evaluation.datagen.question_types import QUESTION_TYPES, get_prompt


class QuestionGenerator:
    """Generate Q&A pairs from document chunks using LLM."""

    def __init__(
        self,
        llm_client: Any,
        question_types: List[str] = None,
        questions_per_chunk: int = 2,
        max_concurrent: int = 5,
    ):
        """Initialize question generator.

        Args:
            llm_client: LangChain LLM client for generation
            question_types: List of question types to generate
            questions_per_chunk: Number of questions per chunk
            max_concurrent: Maximum concurrent LLM calls
        """
        self.llm = llm_client
        self.question_types = question_types or ["factual", "procedural", "comparison"]
        self.questions_per_chunk = questions_per_chunk
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def generate_from_chunk(
        self,
        chunk_id: str,
        content: str,
        question_type: str,
        num_questions: int = 2,
        metadata: Dict[str, Any] = None,
    ) -> List[GeneratedQA]:
        """Generate Q&A pairs from a single chunk.

        Args:
            chunk_id: The chunk identifier
            content: The chunk content
            question_type: Type of questions to generate
            num_questions: Number of questions to generate
            metadata: Optional metadata to include

        Returns:
            List of GeneratedQA objects
        """
        if question_type not in QUESTION_TYPES:
            raise ValueError(f"Unknown question type: {question_type}")

        prompt = get_prompt(question_type, content, chunk_id, num_questions)

        async with self._semaphore:
            try:
                response = await self.llm.ainvoke(prompt)
                response_text = (
                    response.content if hasattr(response, "content") else str(response)
                )
                return self._parse_qa_pairs(
                    response_text, chunk_id, content, question_type, metadata
                )
            except Exception as e:
                print(f"Error generating questions for chunk {chunk_id}: {e}")
                return []

    def _parse_qa_pairs(
        self,
        response: str,
        chunk_id: str,
        content: str,
        question_type: str,
        metadata: Dict[str, Any] = None,
    ) -> List[GeneratedQA]:
        """Parse LLM response into GeneratedQA objects.

        Args:
            response: Raw LLM response text
            chunk_id: Source chunk ID
            content: Source chunk content
            question_type: Type of questions
            metadata: Optional metadata

        Returns:
            List of parsed GeneratedQA objects
        """
        qa_pairs = []

        # Try to extract JSON from response
        json_match = re.search(r"\[[\s\S]*\]", response)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                for item in parsed:
                    if "question" in item and "answer" in item:
                        qa_pairs.append(
                            GeneratedQA(
                                question=item["question"].strip(),
                                answer=item["answer"].strip(),
                                question_type=question_type,
                                source_chunk_ids=item.get("chunk_ids", [chunk_id]),
                                source_content=content,
                                metadata=metadata or {},
                            )
                        )
                return qa_pairs
            except json.JSONDecodeError:
                pass

        # Fallback: try to parse Q:/A: format
        qa_pattern = re.compile(
            r"(?:Q:|Question:)\s*(.+?)\s*(?:A:|Answer:)\s*(.+?)(?=(?:Q:|Question:)|$)",
            re.DOTALL | re.IGNORECASE,
        )

        for match in qa_pattern.finditer(response):
            question = match.group(1).strip()
            answer = match.group(2).strip()
            if question and answer:
                qa_pairs.append(
                    GeneratedQA(
                        question=question,
                        answer=answer,
                        question_type=question_type,
                        source_chunk_ids=[chunk_id],
                        source_content=content,
                        metadata=metadata or {},
                    )
                )

        return qa_pairs

    async def generate_dataset(
        self,
        chunks: List[Dict[str, Any]],
        questions_per_type: int = 20,
        progress_callback: callable = None,
    ) -> EvaluationDataset:
        """Generate full evaluation dataset from chunks.

        Args:
            chunks: List of chunk dicts with 'id', 'content', 'metadata'
            questions_per_type: Target number of questions per type
            progress_callback: Optional callback(current, total, message)

        Returns:
            EvaluationDataset with generated Q&A pairs
        """
        all_qa_pairs = []
        total_chunks = len(chunks)

        if total_chunks == 0:
            print("Warning: No chunks provided for question generation")
            return EvaluationDataset([])

        # Sample chunks for each question type
        for qtype in self.question_types:
            if progress_callback:
                progress_callback(
                    0, questions_per_type, f"Generating {qtype} questions..."
                )

            # Shuffle chunks for variety
            sampled_chunks = chunks.copy()
            random.shuffle(sampled_chunks)

            type_qa_pairs = []
            chunk_idx = 0

            while len(type_qa_pairs) < questions_per_type and chunk_idx < len(
                sampled_chunks
            ):
                chunk = sampled_chunks[chunk_idx]
                chunk_id = chunk.get("id") or chunk.get("chunk_id", f"chunk_{chunk_idx}")
                content = chunk.get("content") or chunk.get("page_content", "")

                if len(content) < 100:
                    chunk_idx += 1
                    continue

                pairs = await self.generate_from_chunk(
                    chunk_id=chunk_id,
                    content=content,
                    question_type=qtype,
                    num_questions=self.questions_per_chunk,
                    metadata=chunk.get("metadata", {}),
                )

                type_qa_pairs.extend(pairs)
                chunk_idx += 1

                if progress_callback:
                    progress_callback(
                        len(type_qa_pairs),
                        questions_per_type,
                        f"Generated {len(type_qa_pairs)}/{questions_per_type} {qtype} questions",
                    )

                # Small delay to avoid rate limiting
                await asyncio.sleep(0.5)

            # Limit to target
            type_qa_pairs = type_qa_pairs[:questions_per_type]
            all_qa_pairs.extend(type_qa_pairs)

            print(f"Generated {len(type_qa_pairs)} {qtype} questions")

        # Deduplicate by question
        seen_questions = set()
        unique_pairs = []
        for qa in all_qa_pairs:
            normalized = qa.question.lower().strip()
            if normalized not in seen_questions:
                seen_questions.add(normalized)
                unique_pairs.append(qa)

        print(f"Total unique Q&A pairs: {len(unique_pairs)}")
        return EvaluationDataset(unique_pairs)

    async def generate_from_rag_client(
        self,
        rag_client: Any,
        questions_per_type: int = 20,
        progress_callback: callable = None,
    ) -> EvaluationDataset:
        """Generate dataset by fetching chunks from RAG service.

        Args:
            rag_client: RAGClient instance
            questions_per_type: Target questions per type
            progress_callback: Optional progress callback

        Returns:
            EvaluationDataset with generated Q&A pairs
        """
        print("Fetching chunks from RAG service...")
        chunk_infos = await rag_client.get_all_chunks()

        if not chunk_infos:
            print("Warning: No chunks retrieved from RAG service")
            return EvaluationDataset([])

        chunks = [
            {
                "id": ci.id,
                "content": ci.content,
                "metadata": ci.metadata,
            }
            for ci in chunk_infos
        ]

        print(f"Retrieved {len(chunks)} chunks")
        return await self.generate_dataset(
            chunks, questions_per_type, progress_callback
        )


def create_generator_from_openai(
    model: str = "gpt-4.1-mini",
    api_key: str = None,
    **kwargs,
) -> QuestionGenerator:
    """Create a QuestionGenerator using OpenAI.

    Args:
        model: OpenAI model name
        api_key: Optional API key (uses env var if not provided)
        **kwargs: Additional args for QuestionGenerator

    Returns:
        Configured QuestionGenerator
    """
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=0.7,
    )

    return QuestionGenerator(llm, **kwargs)
