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


if __name__ == "__main__":
    tok = build_tokenizer()
    demo = tok.tokenize(
        ["Hello WORLD", "What does VLLM_RPC_TIMEOUT default to?"],
        return_as="string",
        show_progress=False,
    )
    print("=== [STEP 3 DIAGNOSTIC] Testing Minimal Tokenizer ===")
    for text, tokens in zip(
        ["Hello WORLD", "What does VLLM_RPC_TIMEOUT default to?"], demo
    ):
        print(f"[Query] {text}")
        print(f"  -> {tokens} ({len(tokens)} tokens)")
    assert demo[0] == ["hello", "world"]
    assert "vllm_rpc_timeout" in demo[1]
    print("[OK] Minimal tokenizer checks passed!")
