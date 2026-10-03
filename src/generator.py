"""Grounded answer generation with Qwen3-0.6B (Step 8).

Context chunks are cited as [Source i] with exact offsets.
Diagnostic builds prompt only, without loading model weights.
"""

from typing import Any, Dict, List, Optional

from src.chunking import Chunk

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"

SYSTEM_PROMPT = (
    "Answer concisely and technically, using only the provided context. "
    "If the context lacks the answer, say so directly."
)


def format_context(chunks: List[Chunk]) -> str:
    """Format chunks as numbered source blocks.

    Args:
        chunks: Retrieved chunks in rank order.

    Returns:
        Multiline string with [Source i] path (start-end) plus text.
    """
    lines: List[str] = []
    for i, ch in enumerate(chunks, 1):
        lines.append(
            f"[Source {i}] {ch.file_path} "
            f"({ch.first_character_index}-{ch.last_character_index}):\n{ch.text}"
        )
    return "\n\n".join(lines)


def build_prompt(question: str, context: str, tokenizer=None) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    if tokenizer and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"


class AnswerGenerator:
    """Lazy Qwen generator with greedy deterministic decoding."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        """Init with model name, weights loaded on first generate().

        Args:
            model_name: HuggingFace model id.
        """
        self.model_name = model_name
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None

    def _ensure_loaded(self) -> None:
        """Load tokenizer and weights once (fp16 on CUDA, fp32 on CPU)."""
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, dtype=dtype
        )
        self._model.eval()

    def generate(
        self, question: str,
        chunks: List[Chunk],
        max_new_tokens: int = 256
    ) -> str:
        """Generate grounded answer from retrieved chunks.

        Args:
            question: User question string.
            chunks: Retrieved top-k chunks as context.
            max_new_tokens: Generation budget.

        Returns:
            Decoded answer string without prompt echo.
        """
        self._ensure_loaded()
        assert self._model is not None and self._tokenizer is not None

        context = format_context(chunks)

        # 1. Use chat template if available to guide the model cleanly
        if hasattr(self._tokenizer, "chat_template") and self._tokenizer.chat_template:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ]
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = build_prompt(question, context)

        # 2. Tokenize and place input tensors on the model device
        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        # 3. Generate greedily
        out = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        decoded = str(
            self._tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
        ).strip()

        # Strip internal thinking tags (<think>...</think>)
        if "</think>" in decoded:
            decoded = decoded.split("</think>")[-1].strip()

        # Clean up any runaway repetition loops
        for marker in ["\n\nAnswer:", "\nAnswer:", "\n\nQuestion:", "\nQuestion:"]:
            decoded = decoded.split(marker)[0].strip()

        return decoded


# Global cache for the singleton instance
_CACHE: Dict[str, AnswerGenerator] = {}


def get_generator(model_name: str = DEFAULT_MODEL) -> AnswerGenerator:
    """Return shared generator instance (Singleton per model name).

    Args:
        model_name: HuggingFace model id.

    Returns:
        Cached AnswerGenerator instance.
    """
    if model_name not in _CACHE:
        _CACHE[model_name] = AnswerGenerator(model_name)
    return _CACHE[model_name]


if __name__ == "__main__":
    print("=== [STEP 8 DIAGNOSTIC] Testing Answer Generator Prompting ===")
    demo = Chunk(
        "data/raw/vllm-0.10.1/docs/features/lora.md",
        4695,
        5100,
        "The /v1/load_lora_adapter HTTP endpoint dynamically loads LoRA adapters.",
    )
    ctx = format_context([demo])
    print("[Formatted Context Preview]:")
    print(ctx)
    prompt = build_prompt("How to load LoRA adapter dynamically?", ctx)
    assert "[Source 1]" in ctx and "lora.md (4695-5100)" in ctx
    assert "Question:" in prompt and "Answer:" in prompt
    assert get_generator() is get_generator()
    print("[Prompt Template Structure Verified]")
