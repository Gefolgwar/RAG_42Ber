"""Grounded answer generation with Qwen3-0.6B, CPU-only."""

from functools import lru_cache
from typing import Any, List, Optional

from src.chunking import Chunk
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"

SYSTEM_PROMPT = (
    "Use only the provided context. Give a direct answer in 1-2 sentences. "
    "No reasoning, no repetitions, no self-dialog, no tags like \\boxed{}. "
    "If the context lacks the answer, say so directly."
)

MAX_CONTEXT_CHARS = 1800
MAX_PROMPT_TOKENS = 1024

# Non-thinking sampling per Qwen3 model card (greedy causes infinite loops).
GEN_KWARGS = {
    "do_sample": True,  # prevents infinite loops
    "temperature": 0.7,
    "top_p": 0.8,  # selecting from a pool with a total probability of 80%
    "top_k": 20,  # selecting from the top 20 candidates
    "min_p": 0.0,  # same as top_p but for the minimum probability of the candidate pool - NOW OFF
    "repetition_penalty": 1.15,  # penalizes repeating the same tokens
    "no_repeat_ngram_size": 3,  # prevents repeating the same 3-words
}
STOP_STRINGS = ["\n\n", "\nQuestion:", "\nContext:", "So, the answer is:", "Answer:", "\\boxed"]


def format_context(chunks: List[Chunk], budget: int = MAX_CONTEXT_CHARS) -> str:
    """Format chunks as [Source i] blocks, clipped to a char budget."""
    lines, total = [], 0
    for i, ch in enumerate(chunks, 1):
        room = budget - total
        if room <= 0:
            break
        snippet = " ".join(ch.text.split())[:room]
        total += len(snippet)
        lines.append(
            f"[Source {i}] {ch.file_path} "
            f"({ch.first_character_index}-{ch.last_character_index}):\n{snippet}"
        )
    return "\n\n".join(lines)


def build_prompt(question: str, context: str) -> str:
    """Assemble system prompt, context and question."""
    return f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"


def _extractive_fallback(question: str, chunks: List[Chunk], budget: int = 600) -> str:
    """First-chunk snippet when the model cannot load."""
    if not chunks:
        return "The context lacks the answer."
    return " ".join(chunks[0].text.split())[:budget].strip()


class AnswerGenerator:
    """Qwen generator on CPU with non-thinking sampling."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float32,
            low_cpu_mem_usage=True
        )

    def generate(self, question: str, chunks: List[Chunk], max_new_tokens: int = 64) -> str:
        """Generate grounded answer from retrieved chunks on CPU."""
        if not chunks:
            return "The context lacks the answer."
        try:
            self._ensure_loaded()
        except Exception:
            return _extractive_fallback(question, chunks)
        assert self._model is not None and self._tokenizer is not None

        context = format_context(chunks)
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
        try:
            try:
                prompt = self._tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            except TypeError:
                prompt = self._tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
        except Exception:
            prompt = build_prompt(question, context)
        # Making tensors for with size limitation
        inputs = self._tokenizer(
            prompt, return_tensors="pt",
            truncation=True,
            max_length=MAX_PROMPT_TOKENS
        )
        input_len = inputs["input_ids"].shape[1]
        # Generationg answers with the model
        try:
            with torch.inference_mode():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self._tokenizer.eos_token_id,
                    stop_strings=STOP_STRINGS,
                    tokenizer=self._tokenizer,
                    **GEN_KWARGS,
                )
        except Exception:
            return _extractive_fallback(question, chunks)
        return str(self._tokenizer.decode(out[0][input_len:], skip_special_tokens=True)).strip()


@lru_cache(maxsize=4)
def get_generator(model_name: str = DEFAULT_MODEL) -> AnswerGenerator:
    """Return shared generator instance (singleton per model name)."""
    return AnswerGenerator(model_name)
