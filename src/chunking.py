"""Chunking facade: single seam for callers."""

from typing import List

from src.chunk_markdown import chunk_markdown
from src.chunk_python import chunk_python
from src.chunk_types import Chunk

__all__ = ["Chunk", "chunk_python", "chunk_markdown", "chunk_file"]


def chunk_file(path: str, text: str, max_chunk_size: int = 2000) -> List[Chunk]:
    """Split by extension: .py -> AST, else headers."""
    if path.endswith(".py"):
        return chunk_python(path, text, max_chunk_size)
    return chunk_markdown(path, text, max_chunk_size)


if __name__ == "__main__":
    print("=== [STEP 5 DIAGNOSTIC] Testing Python AST & Markdown Splitters ===")
    py_sample = "import os\n\n\ndef foo():\n    x = 1\n    return x\n\n\nprint(foo())\n"
    py_chunks = chunk_python("a.py", py_sample, 100)
    print(f"[Python AST Chunking]: Generated {len(py_chunks)} chunks:")
    for i, ch in enumerate(py_chunks, 1):
        assert py_sample[ch.first_character_index:ch.last_character_index] == ch.text
        assert ch.char_len() <= 100
        preview = ch.text.replace("\n", "\\n")[:40]
        print(f"  Chunk #{i} [{ch.first_character_index}:{ch.last_character_index}]"
              f" ({ch.char_len()} chars): {preview}")
        print("  [Verification]: Direct string slice matches chunk text: OK!")
    md_sample = "Intro line\n\n# Title\ntext under title\n\n## Next\nmore text\n"
    md_chunks = chunk_markdown("b.md", md_sample, 100)
    print(f"[Markdown Header Chunking]: Generated {len(md_chunks)} chunks:")
    for ch in md_chunks:
        assert md_sample[ch.first_character_index:ch.last_character_index] == ch.text
        assert ch.char_len() <= 100
        print("  [Verification]: Direct string slice matches chunk text: OK!")
