"""Single seam for search: cache, BM25, hybrid, spans."""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.cache import QueryCache
from src.chunking import Chunk
from src.index import KnowledgeBase
from src.ingest import read_file_safely
from src.models import MinimalSource
from src.dense import load_dense_cached, rrf_fuse


def err(msg: str) -> None:
    """Print error to stderr."""
    print(f"Error: {msg}", file=sys.stderr)


def load_kb(index_directory: str) -> Optional[KnowledgeBase]:
    """Load index or report error."""
    try:
        return KnowledgeBase.load(index_directory)
    except Exception as exc:
        err(f"cannot load index from '{index_directory}': {exc}")
        return None


def fingerprint(index_directory: str) -> str:
    """Read index fingerprint for cache invalidation."""
    try:
        meta = json.loads(
            (Path(index_directory) / "meta.json").read_text("utf-8")
        )
        return str(meta.get("created_at", ""))
    except (OSError, ValueError):
        return ""


def chunk_to_source(ch: Chunk) -> MinimalSource:
    """Convert Chunk to output model."""
    return MinimalSource(
        file_path=ch.file_path,
        first_character_index=ch.first_character_index,
        last_character_index=ch.last_character_index,
    )


def chunk_key(ch: Chunk) -> str:
    """Dedup key for a chunk."""
    return f"{ch.file_path}:{ch.first_character_index}"


def hybrid_search(
    kb: KnowledgeBase, query: str, k: int, file_type: str, index_dir: str
) -> List[Tuple[Chunk, float]]:
    """Fuse BM25 and dense with RRF, fallback to BM25."""
    bm25_pairs = kb.search_bm25(query, k=k, file_type=file_type)
    try:
        dense = load_dense_cached(str(Path(index_dir) / "dense_index"))
        dense.chunks = kb.chunks
        dense_pairs = dense.search_dense(query, k=k, file_type=file_type)
        if not dense_pairs:
            return bm25_pairs
        by_key: Dict[str, Tuple[Chunk, float]] = {}
        for ch, sc in bm25_pairs + dense_pairs:
            by_key.setdefault(chunk_key(ch), (ch, sc))
        fused = rrf_fuse(
            [[chunk_key(ch) for ch, _ in bm25_pairs],
             [chunk_key(ch) for ch, _ in dense_pairs]],
            k=k,
        )
        return [
            (by_key[key][0], score) for key, score in fused if key in by_key
        ]
    except Exception:
        return bm25_pairs


def search_cached(
    kb: KnowledgeBase,
    query: str,
    k: int,
    file_type: str,
    cache: QueryCache,
    index_dir: str,
    hybrid: bool = False,
) -> List[MinimalSource]:
    """Search with cache, store on miss."""
    # 1 cache exist
    hit = cache.get(query, k, hybrid)
    if hit is not None:
        return hit[: max(k, 0)]
    # 2. Cache MISS
    if hybrid:
        pairs = hybrid_search(kb, query, k, file_type, index_dir)
    else:
        pairs = kb.search_bm25(query, k=k, file_type=file_type)
    sources = [chunk_to_source(ch) for ch, _ in pairs]
    cache.put(query, k, sources, hybrid)
    return sources


_SPAN_CACHE: Dict[str, Dict[Tuple[str, int], Chunk]] = {}


def _span_map(index_directory: str) -> Dict[Tuple[str, int], Chunk]:
    """Load chunk span map once per index directory (CPU fast path)."""
    hit = _SPAN_CACHE.get(index_directory)
    if hit is not None:
        return hit
    try:
        kb = KnowledgeBase.load(index_directory)
        by_span: Dict[Tuple[str, int], Chunk] = {
            (c.file_path, c.first_character_index): c for c in kb.chunks
        }
    except Exception:
        by_span = {}
    _SPAN_CACHE[index_directory] = by_span
    return by_span


def resolve_chunks(
        sources: List[MinimalSource],
        index_directory: str
) -> List[Chunk]:
    """Map sources back to texts via index or disk."""
    by_span = _span_map(index_directory)
    out: List[Chunk] = []
    for src in sources:
        hit = by_span.get((src.file_path, src.first_character_index))
        if (
            hit is not None
            and hit.last_character_index == src.last_character_index
        ):
            out.append(hit)
            continue
        # file_path is verbatim corpus path (data/raw/vllm-0.10.1/...),
        # which is also the disk location relative to project root.
        text = read_file_safely(src.file_path)
        span = text[src.first_character_index:src.last_character_index]
        out.append(Chunk(src.file_path, src.first_character_index,
                         src.last_character_index, span))
    return out
