import json
from pathlib import Path
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from src.chunking import Chunk
from src.file_filter import matches

# Name of the embedding model.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@lru_cache(maxsize=1)
def get_model():
    # Load a fresh embedding model every time.
    model = SentenceTransformer(MODEL_NAME)
    return model


class DenseIndex:

    def __init__(self, chunks=None):
        # Keep the list of text chunks.
        if chunks is None:
            self.chunks = []
        else:
            self.chunks = list(chunks)
        # Empty embedding matrix, one row per chunk.
        self.embeddings = np.zeros((0, 384), dtype=np.float32)

    def build(self, show_progress=True):
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

    def save(self, directory):
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
    def load(cls, directory):
        # Read back what save() wrote.
        root = Path(directory)
        idx = cls([])
        idx.embeddings = np.load(str(root / "embeddings.npy"))
        text = (root / "order.json").read_text(encoding="utf-8")
        raw = json.loads(text)
        # Rebuild Chunk objects from plain dicts.
        chunks = []
        for r in raw:
            chunks.append(Chunk(**r))
        idx.chunks = chunks
        return idx

    def search_dense(self, query, k=10, file_type="all"):
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
        out = []
        for i in ranked:
            if matches(self.chunks[i].file_path, file_type):
                out.append((self.chunks[i], float(sims[i])))
            if len(out) >= k:
                break
        return out


def load_dense_cached(directory):
    idx = DenseIndex.load(directory)
    return idx


def rrf_fuse(ranked_lists, k=10):
    # Merge ranked lists with Reciprocal Rank Fusion.
    # Score is sum of 1 / (60 + rank).
    scores = {}
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
