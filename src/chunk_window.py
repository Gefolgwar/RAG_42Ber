"""Sliding window for oversized spans."""

from typing import List

from src.chunk_types import Chunk


def split_oversized(
        path: str,
        text: str,
        start: int,
        end: int,
        max_size: int
) -> List[Chunk]:
    """Split [start:end) with 10% overlap."""
    out: List[Chunk] = []
    step = max(max_size - max_size // 10, 1)
    pos = start
    while pos < end:
        stop = min(pos + max_size, end)
        out.append(Chunk(path, pos, stop, text[pos:stop]))
        if stop >= end:
            break
        pos += step
    return out


def spans(
    path: str,
    text: str,
    bounds: List[int],
    max_size: int
) -> List[Chunk]:
    """Build chunks from sorted bounds."""
    out: List[Chunk] = []
    # For example, if bounds = [0, 300, 1500],
    # zip will create the pairs: (0, 300) and (300, 1500).
    for start, end in zip(bounds, bounds[1:]):
        if end - start <= max_size:
            out.append(Chunk(path, start, end, text[start:end]))
        else:
            out.extend(split_oversized(path, text, start, end, max_size))
    return out
