"""Dense semantic retriever using sentence-transformers and cosine similarity."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from src.chunking import Chunk
from src.file_filter import matches

# Name of the embedding model.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    # Load a fresh embedding model every time.
    model = SentenceTransformer(MODEL_NAME)
    return model


class DenseIndex:
    chunks: List[Chunk]
    embeddings: NDArray[np.float32]

    def __init__(self, chunks: Optional[Sequence[Chunk]] = None) -> None:
        # Keep the list of text chunks.
        if chunks is None:
            self.chunks = []
        else:
            self.chunks = list(chunks)
        # Empty embedding matrix, one row per chunk.
        self.embeddings = np.zeros((0, 384), dtype=np.float32)

    def build(self, show_progress: bool = True) -> None:
        # Do nothing if there are no chunks.
        if not self.chunks:
            return
        model = get_model()
        # Join file path and text so the path also helps search.
        texts = []
        for c in self.chunks:
            texts.append(c.file_path + " " + c.text)
        # Turn every text into a vector.
        vecs = model.encode(texts, show_progress_bar=show_progress)
        self.embeddings = np.asarray(vecs, dtype=np.float32)

    def save(self, directory: Union[str, Path]) -> None:
        # Create the target folder if needed.
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        # Save the vectors as a numpy file.
        np.save(str(root / "embeddings.npy"), self.embeddings)
        # Save the chunk order as json, so load knows which row is which chunk.
        order = []
        for c in self.chunks:
            order.append(c.__dict__)
        text = json.dumps(order)
        (root / "order.json").write_text(text, encoding="utf-8")

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "DenseIndex":
        # Read back what save() wrote.
        root = Path(directory)
        idx = cls([])
        idx.embeddings = np.asarray(
            np.load(str(root / "embeddings.npy")), dtype=np.float32
        )
        text = (root / "order.json").read_text(encoding="utf-8")
        raw: List[dict[str, Any]] = json.loads(text)
        # Rebuild Chunk objects from plain dicts.
        chunks: List[Chunk] = []
        for r in raw:
            chunks.append(Chunk(**r))
        idx.chunks = chunks
        return idx

    def search_dense(
        self, query: str, k: int = 10, file_type: str = "all"
    ) -> List[Tuple[Chunk, float]]:
        # Guard against bad input.
        if not query:
            return []
        if not query.strip():
            return []
        if k <= 0:
            return []
        if not self.chunks:
            return []
        model = get_model()
        # Turn the query into one vector.
        qv = model.encode([query])
        qv = np.asarray(qv, dtype=np.float32)
        # L2-normalize stored vectors for cosine search.
        mat = self.embeddings
        mn = np.linalg.norm(mat, axis=1, keepdims=True)
        mn[mn == 0] = 1.0
        mat = mat / mn
        # L2-normalize the query vector too.
        qn = np.linalg.norm(qv, axis=1, keepdims=True)
        qn[qn == 0] = 1.0
        qv = qv / qn
        # Cosine similarity is a dot product of normalized vectors.
        sims = (mat @ qv[0]).tolist()
        # Best chunk index first.
        ranked = sorted(range(len(self.chunks)), key=lambda i: sims[i], reverse=True)
        # Keep only wanted file type, stop after k hits.
        out: List[Tuple[Chunk, float]] = []
        for i in ranked:
            if matches(self.chunks[i].file_path, file_type):
                out.append((self.chunks[i], float(sims[i])))
            if len(out) >= k:
                break
        return out


def load_dense_cached(directory: Union[str, Path]) -> DenseIndex:
    idx = DenseIndex.load(directory)
    return idx


def rrf_fuse(
    ranked_lists: Sequence[Sequence[str]], k: int = 10
) -> List[Tuple[str, float]]:
    # Merge ranked lists with Reciprocal Rank Fusion.
    # Score is sum of 1 / (60 + rank).
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        rank = 1
        for key in ranked:
            if key not in scores:
                scores[key] = 0.0
            scores[key] = scores[key] + 1.0 / (60 + rank)
            rank = rank + 1
    # Best score first, cut to k.
    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return items[:k]
