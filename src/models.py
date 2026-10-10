"""Pydantic data models for RAG Against the Machine.

Adheres strictly to Chapter VI.4 specifications.
"""

from enum import Enum
from typing import List
import uuid

from pydantic import BaseModel, ConfigDict, Field


class FileType(str, Enum):
    """File category filter for indexing and retrieval."""

    ALL = "all"
    DOCS = "docs"
    CODE = "code"


class MinimalSource(BaseModel):
    """Represents a single retrieved source chunk span."""

    model_config = ConfigDict(frozen=True)

    file_path: str = Field(
        description="Exact file path relative to repo root (e.g. data/raw/...)",
    )
    first_character_index: int = Field(
        ge=0,
        description="Starting character position (inclusive)",
    )
    last_character_index: int = Field(
        ge=0,
        description="Ending character position (exclusive)",
    )


class UnansweredQuestion(BaseModel):
    """An unanswered question from the dataset."""

    question_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the question",
    )
    question: str = Field(
        description="Question text",
    )


class AnsweredQuestion(UnansweredQuestion):
    """A ground truth question with reference answer and sources."""

    sources: List[MinimalSource] = Field(
        default_factory=list,
        description="Ground truth source locations",
    )
    answer: str = Field(
        description="Reference ground truth answer",
    )


class RagDataset(BaseModel):
    """Dataset container matching evaluation JSON structures."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion] = Field(
        default_factory=list,
        description="List of questions in the dataset",
    )


class MinimalSearchResults(BaseModel):
    """Search results for a single question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource] = Field(
        default_factory=list,
        description="Retrieved source spans",
    )


class MinimalAnswer(MinimalSearchResults):
    """Search results combined with the generated model answer."""

    answer: str = Field(
        description="Generated natural language answer",
    )


class StudentSearchResults(BaseModel):
    """Output model for search operations."""

    search_results: List[MinimalSearchResults] = Field(
        default_factory=list,
        description="List of search results per question",
    )
    k: int = Field(
        ge=0,
        description="Number of results requested",
    )


class StudentSearchResultsAndAnswer(BaseModel):
    """Output model for answer generation operations."""

    search_results: List[MinimalAnswer] = Field(
        default_factory=list,
        description="List of answers with retrieved sources per question",
    )
    k: int = Field(
        ge=0,
        description="Number of results requested",
    )


if __name__ == "__main__":
    print("=== [STEP 2 DIAGNOSTIC] Testing Pydantic Models ===")

    # 1. Test MinimalSource instantiation and immutability
    src1 = MinimalSource(
        file_path="data/raw/vllm-0.10.1/docs/features/lora.md",
        first_character_index=4695,
        last_character_index=6098,
    )
    src2 = MinimalSource(
        file_path="data/raw/vllm-0.10.1/docs/features/lora.md",
        first_character_index=4695,
        last_character_index=6098,
    )
    print(f"[OK] MinimalSource created: {src1.file_path} "
          f"[{src1.first_character_index}:{src1.last_character_index}]")

    # 2. Test deduplication via set
    unique_sources = {src1, src2}
    assert len(unique_sources) == 1, "Deduplication failed!"
    print(f"[OK] Immutability & Set deduplication verified "
          f"(set len = {len(unique_sources)}, expected 1)")

    # 3. Test question creation
    q = UnansweredQuestion(question="How to configure vLLM?")
    print(f"[OK] UnansweredQuestion created: ID={q.question_id}, question='{q.question}'")

    # 4. Test RagDataset with pipe union syntax
    dataset = RagDataset(
        rag_questions=[
            q,
            AnsweredQuestion(
                question="What is LoRA?",
                sources=[src1],
                answer="Low-Rank Adaptation",
            ),
        ]
    )
    assert len(dataset.rag_questions) == 2
    print(f"[OK] RagDataset validated with pipe syntax: {len(dataset.rag_questions)} questions")

    # 5. Test StudentSearchResults serialization
    res = StudentSearchResults(
        search_results=[
            MinimalSearchResults(
                question_id=q.question_id,
                question=q.question,
                retrieved_sources=[src1],
            )
        ],
        k=1,
    )
    json_output = res.model_dump_json(indent=2)
    print("[OK] Serialized StudentSearchResults JSON sample:")
    print(json_output)
