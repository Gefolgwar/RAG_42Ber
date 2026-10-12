"""Minimal tokenizer for the learning RAG skeleton (Step 3).

Learning-first approach: no camelCase/snake_case pre-splitting.
Lowercasing and English stopword removal are delegated to
``bm25s.tokenization.Tokenizer``. Identifiers stay whole
(``VLLM_RPC_TIMEOUT`` -> ``vllm_rpc_timeout``), exactly like
the reference example. Code-aware splitting is a deferred
improvement, not part of the skeleton.
"""

from bm25s.tokenization import Tokenizer

DEFAULT_SPLITTER = r"\w+"


def build_tokenizer() -> Tokenizer:
    return Tokenizer(
        lower=True,
        splitter=DEFAULT_SPLITTER,
        stopwords="english",
        stemmer=None,
    )
