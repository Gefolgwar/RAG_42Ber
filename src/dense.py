"""Dense embeddings and hybrid RRF fusion (Bonus 1 and 2)."""

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt  # avoid mypy erros

from src.chunking import Chunk
from src.file_filter import matches

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RRF_K = 60


@lru_cache(maxsize=1)
def get_model():  # type: ignore[no-untyped-def]
    """Return shared SentenceTransformer instance (loaded once)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def _norm_rows(mat: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """L2-normalize matrix rows, guarding against zero vectors."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    res: npt.NDArray[np.float32] = mat / norms
    return res


class DenseIndex:
    """384-dim MiniLM embeddings with cosine search and npy persistence."""

    def __init__(self, chunks: List[Chunk] | None = None) -> None:
        """Init with chunks and empty embedding matrix.

        Args:
            chunks: Chunks aligned with embedding rows.
        """
        self.chunks: List[Chunk] = list(chunks) if chunks else []
        self.embeddings: npt.NDArray[np.float32] = np.zeros((0, 384), dtype=np.float32)

    def build(self, show_progress: bool = True, batch_size: int = 32) -> None:
        """Encode chunk texts (file path prepended) into embeddings."""
        if not self.chunks:
            return
        from tqdm import tqdm

        tqdm.write(f"Loading dense model '{MODEL_NAME}'...")
        model = get_model()
        texts = [f"{c.file_path} {c.text}" for c in self.chunks]
        tqdm.write(f"Encoding {len(texts)} chunks...")
        vecs = model.encode(
            texts,
            show_progress_bar=show_progress,
            batch_size=batch_size,
            convert_to_numpy=True,
        )
        self.embeddings = np.asarray(vecs, dtype=np.float32)

    def save(self, directory: str) -> None:
        """Save embeddings.npy plus chunk order sidecar.

        Args:
            directory: Target directory (created if missing).
        """
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        np.save(str(root / "embeddings.npy"), self.embeddings)
        import json

        order = [c.__dict__ for c in self.chunks]
        (root / "order.json").write_text(json.dumps(order), encoding="utf-8")

    @classmethod
    def load(cls, directory: str) -> "DenseIndex":
        """Restore embeddings and chunk order from disk.

        Args:
            directory: Directory previously written by save().

        Returns:
            DenseIndex with embeddings and chunks restored.
        """
        import json

        root = Path(directory)
        idx = cls([])
        idx.embeddings = np.load(str(root / "embeddings.npy"))
        raw = json.loads((root / "order.json").read_text(encoding="utf-8"))
        idx.chunks = [Chunk(**r) for r in raw]
        return idx

    def search_dense(
        self, query: str, k: int = 10, file_type: str = "all"
    ) -> List[Tuple[Chunk, float]]:
        """Cosine-similarity search with file-type filtering.

        Args:
            query: User question string.
            k: Number of results to return.
            file_type: "all", "docs" or "code".

        Returns:
            Ranked (chunk, cosine score) pairs.
        """
        if not query or not query.strip() or k <= 0 or not self.chunks:
            return []
        model = get_model()
        qv = np.asarray(model.encode([query], convert_to_numpy=True), dtype=np.float32)
        mat = _norm_rows(self.embeddings)
        qn = _norm_rows(qv)
        sims = (mat @ qn[0]).tolist()
        ranked = sorted(range(len(self.chunks)), key=lambda i: sims[i], reverse=True)
        out: List[Tuple[Chunk, float]] = []
        for i in ranked:
            if matches(self.chunks[i].file_path, file_type):
                out.append((self.chunks[i], float(sims[i])))
            if len(out) >= k:
                break
        return out


@lru_cache(maxsize=4)
def load_dense_cached(directory: str) -> "DenseIndex":
    """Return shared DenseIndex instance (embeddings loaded once)."""
    return DenseIndex.load(directory)


def rrf_fuse(
    ranked_lists: List[List[str]], k: int = 10, rrf_k: int = RRF_K
) -> List[Tuple[str, float]]:
    """Fuse ranked key lists with Reciprocal Rank Fusion.

    Args:
        ranked_lists: Lists of chunk keys in rank order per method.
        k: Number of fused keys to return.
        rrf_k: RRF smoothing constant (60).

    Returns:
        (key, fused score) pairs sorted by descending score.
    """
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, key in enumerate(ranked, 1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: max(k, 0)]
