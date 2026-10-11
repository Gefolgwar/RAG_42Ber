"""BM25 knowledge base with persistence (Step 6).

Chain: chunks -> "file_path + text" -> bm25s -> top-k search.
Layout: chunks.json + meta.json + bm25_index/ under save directory.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Tuple

import bm25s  # type: ignore[import-untyped]

from src.chunking import Chunk
from src.file_filter import matches
from src.tokenizer import build_tokenizer


class KnowledgeBase:
    """Chunk collection with BM25 index, save/load and search."""

    def __init__(self, chunks: Optional[List[Chunk]] = None) -> None:
        """Init with chunks and fresh tokenizer plus retriever.

        Args:
            chunks: List of chunks to index (empty list by default).
        """
        self.chunks: List[Chunk] = list(chunks) if chunks else []
        self.tokenizer = build_tokenizer()
        self.retriever: Any = bm25s.BM25()

    def build(self, show_progress: bool = True) -> None:
        """Index chunk texts enriched with file paths."""
        if not self.chunks:
            return

        # for с in self.chunks[:10]:
        #     print(f"Chunk: {с.file_path} [{с.first_character_index}:{с.last_character_index}]")
        #     print(f"Text: {repr(с.text[:100])}\n")

        texts = [f"{c.file_path} {c.text}" for c in self.chunks]
        tokens = self.tokenizer.tokenize(
            texts, return_as="tuple", show_progress=show_progress
        )
        self.retriever.index(tokens, show_progress=show_progress)

    def save(self, directory: str, show_progress: bool = True) -> None:
        """Persist chunks, meta and BM25 files under directory.

        Args:
            directory: Target directory (created if missing).
            show_progress: Show progress bar while writing index files.
        """
        root = Path(directory)
        idx_dir = root / "bm25_index"
        idx_dir.mkdir(parents=True, exist_ok=True)
        # Save BM25 index
        self.retriever.save(str(idx_dir), show_progress=show_progress)
        # Save vocab
        self.tokenizer.save_vocab(str(idx_dir))
        # Save chunks
        payload = [c.__dict__ for c in self.chunks]
        (root / "chunks.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        meta = {
            "num_chunks": len(self.chunks),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str) -> "KnowledgeBase":
        """Restore index state from disk in milliseconds.

        Args:
            directory: Directory previously written by save().

        Returns:
            KnowledgeBase with chunks, retriever and tokenizer restored.
        """
        root = Path(directory)
        idx_dir = root / "bm25_index"
        kb = cls([])
        # Load BM25 index
        kb.retriever = bm25s.BM25.load(str(idx_dir), load_corpus=False)
        # Load tokenizer vocab
        kb.tokenizer = build_tokenizer()
        # Load chunks
        kb.tokenizer.load_vocab(str(idx_dir))
        raw: List[dict[str, Any]] = json.loads(
            (root / "chunks.json").read_text(encoding="utf-8")
        )
        kb.chunks = [Chunk(**item) for item in raw]
        return kb

    def search_bm25(
        self, query: str, k: int = 10, file_type: str = "all"
    ) -> List[Tuple[Chunk, float]]:
        """Lexical search with optional file-type filtering.

        Args:
            query: User question string.
            k: Number of results to return.
            file_type: "all", "docs" or "code".

        Returns:
            Ranked (chunk, score) pairs, empty on degenerate input.
        """
        if not query or not query.strip() or k <= 0 or not self.chunks:
            return []
        pool = min(len(self.chunks), max(k * 3, 50))
        qtokens = self.tokenizer.tokenize([query], return_as="tuple")
        ids, scores = self.retriever.retrieve(qtokens, k=pool)
        out: List[Tuple[Chunk, float]] = []
        for idx, score in zip(ids[0].tolist(), scores[0].tolist()):
            chunk = self.chunks[int(idx)]
            if matches(chunk.file_path, file_type):
                out.append((chunk, float(score)))
            if len(out) >= k:
                break
        return out
