*This project has been created as part of the 42 curriculum by gefolgwar.*

# RAG Against the Machine

## Description

Retrieval-Augmented Generation system that answers questions about the vLLM
codebase. The pipeline ingests the vLLM repository into a searchable index,
retrieves the most relevant snippets for a question with lexical BM25
(optionally fused with dense embeddings), generates a grounded answer with
the local `Qwen/Qwen3-0.6B` model, and measures retrieval quality with
`recall@k` (IoU >= 0.05 on exact file paths).

Goal: reach at least 80% Recall@5 on docs questions and 50% on code
questions, with indexing under 5 minutes and 200-question retrieval under
90 seconds.


## Diagram
[![Architecture diagram](https://gitdiagram.com/diagram-badge.svg)](https://gitdiagram.com/gefolgwar/rag_42ber?utm_source=readme&utm_medium=badge)

[![Architecture diagram of gefolgwar/rag_42ber](https://gitdiagram.com/gefolgwar/rag_42ber/diagram.png)](https://gitdiagram.com/gefolgwar/rag_42ber?utm_source=readme&utm_medium=picture)

## System architecture

```text
data/raw/vllm-0.10.1
  -> ingest (scan + safe UTF-8 read)
  -> chunking (Python AST / Markdown headers, max 2000 chars)
  -> index (BM25s + file-path enrichment) -> data/processed/
  -> retrieve (top-k, file-type filter, query cache)
  -> augment (numbered [Source i] context blocks)
  -> generate (Qwen3-0.6B, sampling) -> JSON answer
  -> evaluate (Recall@1/3/5/10 vs ground truth)
```

Components (`src/`, each 50-150 lines):

| Module | Role |
|---|---|
| `models.py` | Pydantic schemas (Chapter VI.4) |
| `tokenizer.py` | Minimal `bm25s` tokenizer (lower, `\w+`, english stopwords) |
| `ingest.py` | Recursive scan, extension whitelist, UTF-8 `errors="replace"` |
| `chunking.py` | AST chunking for `.py`, header chunking for docs |
| `index.py` | `KnowledgeBase`: build/save/load BM25, `search_bm25` |
| `evaluator.py` | IoU overlap, Recall@k, `EvalReport` table |
| `generator.py` | Qwen3-0.6B singleton, grounded prompt builder |
| `cli.py` | 6 Fire commands (`__main__.py` entry: `python -m src`) |
| `hash.py` | Bonus 3: SHA-256 incremental indexing |
| `cache.py` | Bonus 4: disk query cache with index fingerprint |
| `dense.py` | Bonus 1+2: MiniLM embeddings + RRF hybrid fusion |

`api/app.py` (Bonus 5) exposes `POST /search` and `POST /answer` via FastAPI.

## Chunking strategy

- **Python** (`chunk_python`): `ast.parse` top-level `FunctionDef` /
  `AsyncFunctionDef` / `ClassDef` boundaries (decorators included); imports
  and globals before the first def become a preamble chunk, code after the
  last def becomes a tail chunk; `SyntaxError` falls back to plain windowing.
- **Markdown/text** (`chunk_markdown`): split on ATX headers (`#`..`######`);
  leading intro block starts at offset 0.
- **Invariant**: `content[start:end] == chunk.text` for every chunk.
- **Oversized spans** (`_split_oversized`): sliding window of
  `max_chunk_size` (default 2000, hard limit per Moulinette
  `max_context_length`) with 10% overlap.

## Retrieval method

- **Lexical (mandatory)**: BM25s over `"{file_path} {chunk.text}"` so queries
  naming modules (`sampler.py`, `lora.md`) still match. Query tokenized with
  the same minimal tokenizer. `file_type` filter (`all`/`docs`/`code`) pulls
  an extended pool `min(N, max(k*3, 50))` then filters by extension.
- **Dense (Bonus 1)**: `all-MiniLM-L6-v2` 384-dim embeddings of the same
  enriched texts, cosine search, persisted as `dense_index/embeddings.npy`.
- **Hybrid (Bonus 2)**: Reciprocal Rank Fusion
  `score(d) = sum(1 / (60 + rank))` over BM25 and dense lists.
- **Cache (Bonus 4)**: SHA-256 of `query.lower():k`, invalidated when the
  index fingerprint (`meta.json created_at`) changes.

## Performance analysis

Measured on this machine (corpus: 1952 files -> 28,837 chunks):

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---|---|---|---|---|
| docs (100 q) | 0.610 | 0.750 | **0.810 (PASS >= 80%)** | 0.870 |
| code (99 q) | 0.333 | 0.495 | **0.545 (PASS >= 50%)** | 0.606 |

- Indexing: ~10 s lexical (limit: 5 min). Dense CPU encoding is opt-in
  (`index --dense True`) because 28k MiniLM vectors exceed the budget.
- Retrieval: ~140 questions/s (limit: 90 s per 200 questions).
- All retrieved spans <= 2000 chars (`max_context_length` valid).

## Design decisions

- **No heavy frameworks** (no LangChain, no Ollama daemon): direct `ast`
  parsing, header splitting and `bm25s` keep the pipeline reproducible under
  Moulinette's isolated check.
- **Frozen Pydantic sources**: `MinimalSource(frozen=True)` enables `set()`
  deduplication and prevents span mutation.
- **Union order**: `RagDataset` lists `AnsweredQuestion` before
  `UnansweredQuestion` so Pydantic never truncates `sources`/`answer`.
- **Incremental (Bonus 3)**: SHA-256 per file in `hashes.json`; unchanged
  runs print `Index up-to-date!`; changed files are re-chunked while kept
  chunks are reused (BM25 statistics are refit globally, which is required
  for correct scoring).
- **Sampling generation**: non-thinking sampling per Qwen3 model card
  (`do_sample=True`, `temperature=0.7`) with repetition penalty and
  extractive fallback; grading prioritizes grounding over phrasing.

## Challenges faced

1. **Verbatim `file_path` matching**: grader compares paths exactly
   (`data/raw/vllm-0.10.1/...`). Solved by collecting with
   `relative_to="."`, so logical paths equal disk paths — no mapping
   helpers needed.
2. **Lint scope**: `mypy .` / `flake8 .` scanned the vendored corpus and
   failed; `Makefile` lint now targets `src api` with `max-line-length=100`.
3. **Dense cost**: 28k-vector CPU encoding threatened the 5-minute budget,
   so it is opt-in; lexical alone passes both recall bars.
4. **0.6B repetition loops** (greedy decoding causes infinite loops per
   Qwen3 model card): solved with sampling (`do_sample=True`,
   `repetition_penalty=1.15`) plus an extractive fallback; answers stay
   on-topic and cite context.

## Instructions

Prerequisites: Python 3.10+, `uv`.

```bash
make install   # uv sync
make run       # index with defaults (max_chunk_size=2000)
make lint      # flake8 + mypy (subject flags)
make clean     # drop caches and generated indices/outputs
```

Full pipeline (mirrors the reference exam scripts):

```bash
uv run python -m src index --max_chunk_size 2000
uv run python -m src search "How to configure OpenAI server?" --k 2
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions
uv run python -m src evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
uv run python -m src evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json
uv run python -m src answer "What HTTP endpoint loads a LoRA adapter?" --k 2
uv run python -m src answer_dataset --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --save_directory data/output/search_results_and_answer/UnansweredQuestions
uv run python -m src answer_dataset --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json --save_directory data/output/search_results_and_answer/UnansweredQuestions
make api       # Bonus 5: FastAPI on :8000 (/search, /answer)
# or
uv run uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
# than
$ curl -X POST localhost:8000/search -H "Content-Type: application/json" -d '{"query": "How to configure LoRA?", "k": 2}'
```

## Example usage

```bash
$ uv run python -m src search "What HTTP endpoint is used to dynamically load a LoRA adapter?" --k 1
[{"file_path": "data/raw/vllm-0.10.1/docs/features/lora.md",
  "first_character_index": 4695, "last_character_index": 6100}]

$ curl -X POST localhost:8000/search -H "Content-Type: application/json" -d '{"query": "How to configure LoRA?", "k": 2}'
```

## Resources

- vLLM documentation (indexed corpus itself, `data/raw/vllm-0.10.1/docs/`)
- BM25 (Robertson & Zaragoza) + `bm25s` library docs
- Sentence-Transformers `all-MiniLM-L6-v2` model card
- Qwen3-0.6B model card (Qwen)
- FastAPI documentation; Python Fire documentation
- 42 subject `docs/en.subject.pdf` + `docs/step_by_step_specification.md`
- Hybrid Search Explained (Schovia) — https://www.youtube.com/watch?v=_i_69xycMOw&t=284s
- Retrieval (Schovia) — https://www.youtube.com/watch?v=LB_MFYCjb3U
- karpatiy — https://www.youtube.com/watch?v=zduSFxRajkE&t=2103s
- RAG Full Course in Python — https://www.youtube.com/watch?v=qoQCpOSqNeI
- py RAG tutorial (pixegami) — https://www.youtube.com/watch?v=2TJxpyO3ei4
- RAG + Langchain Python Project (pixegami) — https://www.youtube.com/watch?v=tcqEUSNCn8I
- Hugging Face RAG evaluation cookbook — https://huggingface.co/learn/cookbook/en/rag_evaluation
- Guide to End-to-End RAG Systems Evaluation — https://www.scribd.com/document/789686983/Guide-to-End-to-End-RAG-Systems-Evaluation

### AI usage

AI assistance (code-oriented LLM agent) was used for the following tasks
and parts: scaffolding module skeletons (`src/models.py`, `src/cli.py`,
`api/app.py`) from the step-by-step specification, drafting docstrings and
the README structure (`Description`, `System architecture`, `Resources`),
and diagnosing errors in path handling (`src/ingest.py`, `src/index.py`)
and lint scope (`Makefile`). All retrieval thresholds were verified by
executing the real pipeline (index -> search_dataset -> evaluate); every
metric quoted above comes from actual runs. Final code was reviewed,
trimmed to minimal size, and kept under human control per the 42 AI
instructions.
