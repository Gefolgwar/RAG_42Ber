"""Markdown splitter: one chunk per header section."""

from typing import List

from src.chunk_types import Chunk
from src.chunk_window import spans


def is_header(line: str) -> bool:
    """True for ATX headers (# .. ######)."""
    stripped = line.lstrip()
    if not stripped.startswith("#"):
        return False
    hashes = len(stripped) - len(stripped.lstrip("#"))
    if hashes < 1 or hashes > 6:
        return False
    if len(stripped) == hashes:
        return True
    return stripped[hashes] in " \t\r\n"


def chunk_markdown(path: str, text: str, max_chunk_size: int = 2000) -> List[Chunk]:
    """Split by headers."""
    bounds = [0]
    pos = 0
    for line in text.splitlines(keepends=True):
        if pos > 0 and is_header(line):
            bounds.append(pos)
        pos += len(line)
    bounds.append(len(text))
    return spans(path, text, sorted(set(bounds)), max_chunk_size)
