"""AST and header chunking with exact character offsets (Step 5).

Python files split by top-level def/class, Markdown by ATX headers.
Main invariant: file_content[first:last] == chunk.text.
"""

import ast
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    """Single searchable span with exact file offsets."""

    file_path: str
    first_character_index: int
    last_character_index: int
    text: str

    def char_len(self) -> int:
        """Return chunk length in characters."""
        return self.last_character_index - self.first_character_index


def _split_oversized(
    path: str, text: str, start: int, end: int, max_chunk_size: int
) -> List[Chunk]:
    """Split [start:end) with sliding window and 10% overlap.

    Args:
        path: Relative file path for resulting chunks.
        text: Full file content.
        start: Inclusive start offset.
        end: Exclusive end offset.
        max_chunk_size: Maximum chunk length in characters.

    Returns:
        List of chunks covering [start:end).
    """
    overlap = max_chunk_size // 10
    step = max(max_chunk_size - overlap, 1)
    out: List[Chunk] = []
    pos = start
    while pos < end:
        stop = min(pos + max_chunk_size, end)
        out.append(Chunk(path, pos, stop, text[pos:stop]))
        if stop >= end:
            break
        pos += step
    return out


def _spans(path: str, text: str, bounds: List[int], max_chunk_size: int) -> List[Chunk]:
    """Build chunks from sorted bounds, splitting oversized spans.

    Args:
        path: Relative file path for resulting chunks.
        text: Full file content.
        bounds: Sorted unique offsets including 0 and len(text).
        max_chunk_size: Maximum chunk length in characters.

    Returns:
        List of chunks with exact slices.
    """
    out: List[Chunk] = []
    for start, end in zip(bounds, bounds[1:]):
        if end - start <= max_chunk_size:
            out.append(Chunk(path, start, end, text[start:end]))
        else:
            out.extend(_split_oversized(path, text, start, end, max_chunk_size))
    return out


def chunk_python(path: str, text: str, max_chunk_size: int = 2000) -> List[Chunk]:
    """Split Python source by top-level defs/classes via AST.

    Args:
        path: Relative file path.
        text: File content.
        max_chunk_size: Maximum chunk length.

    Returns:
        Chunks for preamble, each def/class, and file tail.
        Falls back to plain windowing on SyntaxError.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _split_oversized(path, text, 0, len(text), max_chunk_size)
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))

    def off(lineno: int, col: int) -> int:
        """Convert (lineno, col) to global character index."""
        return offsets[lineno - 1] + col

    bounds = [0]
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            lines = [node.lineno] + [d.lineno for d in node.decorator_list]
            start = off(min(lines), 0)
            end = off(node.end_lineno or node.lineno, node.end_col_offset or 0)
            if start > bounds[-1]:
                bounds.append(start)
            bounds.append(end)
    bounds.append(len(text))
    return _spans(path, text, sorted(set(bounds)), max_chunk_size)


def chunk_markdown(path: str, text: str, max_chunk_size: int = 2000) -> List[Chunk]:
    """Split Markdown/text by ATX headers (# .. ######).

    Args:
        path: Relative file path.
        text: File content.
        max_chunk_size: Maximum chunk length.

    Returns:
        Chunks for intro block and each header section.
    """
    bounds = [0]
    lines = text.splitlines(keepends=True)
    pos = 0
    for line in lines:
        stripped = line.lstrip()
        hashes = len(stripped) - len(stripped.lstrip("#"))
        is_header = 1 <= hashes <= 6 and (
            len(stripped) == hashes or stripped[hashes] in " \t\r\n"
        )
        if is_header and pos > 0:
            bounds.append(pos)
        pos += len(line)
    bounds.append(len(text))
    return _spans(path, text, sorted(set(bounds)), max_chunk_size)


def chunk_file(path: str, text: str, max_chunk_size: int = 2000) -> List[Chunk]:
    """Dispatch to Python or Markdown splitter by extension.

    Args:
        path: Relative file path.
        text: File content.
        max_chunk_size: Maximum chunk length.

    Returns:
        List of chunks from the matching splitter.
    """
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
