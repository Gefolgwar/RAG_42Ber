"""BM25 knowledge base with persistence (Step 6).

Chain: chunks -> "file_path + text" -> bm25s -> top-k search.
Layout: chunks.json + meta.json + bm25_index/ under save directory.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Tuple

import bm25s

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

    def build(self) -> None:
        """Index chunk texts enriched with file paths."""
        if not self.chunks:
            return
        texts = [f"{c.file_path} {c.text}" for c in self.chunks]
        tokens = self.tokenizer.tokenize(texts, return_as="tuple")
        self.retriever.index(tokens)

    def save(self, directory: str) -> None:
        """Persist chunks, meta and BM25 files under directory.

        Args:
            directory: Target directory (created if missing).
        """
        root = Path(directory)
        idx_dir = root / "bm25_index"
        idx_dir.mkdir(parents=True, exist_ok=True)
        self.retriever.save(str(idx_dir), show_progress=False)
        self.tokenizer.save_vocab(str(idx_dir))
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
        kb.retriever = bm25s.BM25.load(str(idx_dir), load_corpus=False)
        kb.tokenizer = build_tokenizer()
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


if __name__ == "__main__":
    import shutil

    print("=== [STEP 6 DIAGNOSTIC] Testing BM25s Indexing & Search ===")
    demo = [
        Chunk("docs/lora.md", 100, 300, "load LoRA adapter weights dynamically"),
        Chunk("vllm/entrypoints/api.py", 50, 200, "entrypoint server code"),
        Chunk("docs/other.md", 0, 50, "unrelated cooking recipe"),
    ]
    kb = KnowledgeBase(demo)
    kb.build()
    test_dir = "data/processed/test_index"
    kb.save(test_dir)
    print(f"[Index] Saved index and {len(demo)} chunks to {test_dir}/")
    kb2 = KnowledgeBase.load(test_dir)
    print(f"[OK] Index reloaded successfully. Total chunks: {len(kb2.chunks)}")
    q = "How to load LoRA adapter dynamically?"
    print(f"[Search Query]: '{q}'")
    hits = kb2.search_bm25(q, k=2)
    assert hits and hits[0][0].file_path == "docs/lora.md"
    for rank, (ch, sc) in enumerate(hits, 1):
        print(f"  Rank #{rank} (Score: {sc:.2f}): "
              f"Path: {ch.file_path} [{ch.first_character_index}:"
              f"{ch.last_character_index}]")
    shutil.rmtree(test_dir, ignore_errors=True)
