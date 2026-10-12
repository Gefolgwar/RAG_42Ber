"""Chunking facade: single seam for callers."""

from typing import List

from src.chunk_markdown import chunk_markdown
from src.chunk_python import chunk_python
from src.chunk_types import Chunk

__all__ = ["Chunk", "chunk_python", "chunk_markdown", "chunk_file"]


def chunk_file(
        path: str,
        text: str,
        max_chunk_size: int = 2000
) -> List[Chunk]:
    """Split by extension: .py -> AST, else headers."""
    if path.endswith(".py"):
        return chunk_python(path, text, max_chunk_size)
    return chunk_markdown(path, text, max_chunk_size)
