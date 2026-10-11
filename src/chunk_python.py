"""Python splitter: one chunk per top def/class."""

import ast
from typing import List

from src.chunk_types import Chunk
from src.chunk_window import spans, split_oversized


def line_offsets(text: str) -> List[int]:
    """Start index of each line."""
    out = [0]
    for line in text.splitlines(keepends=True):
        out.append(out[-1] + len(line))
    return out


def chunk_python(path: str, text: str, max_chunk_size: int = 2000) -> List[Chunk]:
    """Split by top defs/classes, fallback to window."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return split_oversized(path, text, 0, len(text), max_chunk_size)
    offsets = line_offsets(text)

    def pos(lineno: int, col: int) -> int:
        return offsets[lineno - 1] + col

    # bounds list of indexis, where file spilts, for example bounds = [0, 150, 420, 600, 1200]
    bounds = [0]
    for node in tree.body:  # tree.body - list of all instractions
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        # start
        lines = [node.lineno] + [d.lineno for d in node.decorator_list]
        start = pos(min(lines), 0)
        # end
        end = pos(node.end_lineno or node.lineno, node.end_col_offset or 0)
        if start > bounds[-1]:
            bounds.append(start)
        bounds.append(end)
    bounds.append(len(text))
    return spans(path, text, sorted(set(bounds)), max_chunk_size)
