*This project has been created as part of the 42 curriculum by <login>.*

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

## System Architecture

```text
data/raw/vllm-0.10.1
  -> ingest (scan + safe UTF-8 read)
  -> chunking (Python AST / Markdown headers, max 2000 chars)
  -> index (BM25s + file-path enrichment) -> data/processed/
  -> retrieve (top-k, file-type filter, query cache)
  -> augment (numbered [Source i] context blocks)
  -> generate (Qwen3-0.6B, greedy) -> JSON answer
  -> evaluate (Recall@1/3/5/10 vs ground truth)
```

Components (`src/`, each 50-150 lines with `__main__` diagnostics):

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

## Chunking Strategy

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

## Retrieval Method

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

## Performance Analysis

Measured on this machine (corpus: 1952 files -> 28,837 chunks):

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---|---|---|---|---|
| docs (100 q) | 0.610 | 0.750 | **0.810 (PASS >= 80%)** | 0.870 |
| code (99 q) | 0.333 | 0.495 | **0.545 (PASS >= 50%)** | 0.606 |

- Indexing: ~10 s lexical (limit: 5 min). Dense CPU encoding is opt-in
  (`index --dense True`) because 28k MiniLM vectors exceed the budget.
- Retrieval: ~140 questions/s (limit: 90 s per 200 questions).
- All retrieved spans <= 2000 chars (`max_context_length` valid).

## Design Decisions

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
- **Deterministic generation**: greedy decoding (`do_sample=False`) for
  reproducible answers; grading prioritizes grounding over phrasing.

## Challenges Faced

1. **Verbatim `file_path` matching**: grader compares paths exactly
   (`data/raw/vllm-0.10.1/...`). Solved by collecting with
   `relative_to="."`, so logical paths equal disk paths — no mapping
   helpers needed.
2. **Lint scope**: `mypy .` / `flake8 .` scanned the vendored corpus and
   failed; `Makefile` lint now targets `src api` with `max-line-length=100`.
3. **Dense cost**: 28k-vector CPU encoding threatened the 5-minute budget,
   so it is opt-in; lexical alone passes both recall bars.
4. **0.6B repetition loops** under greedy decoding: accepted per subject
   (grounding over phrasing); answers stay on-topic and cite context.

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
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 --save_directory data/output/search_results/UnansweredQuestions
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
uv run python -m src answer "What HTTP endpoint loads a LoRA adapter?" --k 2
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
make api       # Bonus 5: FastAPI on :8000 (/search, /answer)
```

Module diagnostics: `uv run python -m src.models|tokenizer|ingest|chunking|`
`index|evaluator|generator` (each prints its `[STEP N DIAGNOSTIC]` block).

## Example Usage

```bash
$ uv run python -m src search "What HTTP endpoint is used to dynamically load a LoRA adapter?" --k 1
[{"file_path": "data/raw/vllm-0.10.1/docs/features/lora.md",
  "first_character_index": 4695, "last_character_index": 6100}]

$ curl -X POST localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "How to configure LoRA?", "k": 2}'
```

## Testing Plan

This is the exact verification sequence used to validate the project, in the
order a reviewer should run it. Every step below was executed; expected
outputs are quoted from real runs.

### Phase 0 — Environment

```bash
make install   # uv sync, the only command the reviewer/moulinette runs
uv run python -c "import pydantic, fire, tqdm, bm25s, torch, transformers; \
print('Dependencies OK! Torch version:', torch.__version__)"
```

Watch: `uv sync` must succeed from the repo root (`pyproject.toml` +
`uv.lock` present). First Qwen/MiniLM downloads need network + disk space
(weights are never committed, see `.gitignore`).

### Phase 1 — Module diagnostics (bottom-up)

Each module self-tests via `__main__`; run them in pipeline order:

| # | Command | What it proves | Expected output |
|---|---|---|---|
| 2 | `uv run python -m src.models` | Pydantic schemas, `frozen=True` dedup | `[OK] MinimalSource created`, `set len = 1` |
| 3 | `uv run python -m src.tokenizer` | lower + stopwords, identifiers whole | `vllm_rpc_timeout` kept intact |
| 4 | `uv run python -m src.ingest` | corpus scan (>1000 files expected) | `[Summary] Total files discovered: 1952` |
| 5 | `uv run python -m src.chunking` | AST + header splits, slice invariant | 3 Python + 3 Markdown chunks, all `OK!` |
| 6 | `uv run python -m src.index` | BM25 save/load round-trip | `Rank #1 (Score: 1.64): Path: docs/lora.md` |
| 7 | `uv run python -m src.evaluator` | IoU math, path matching | `0.50`, `0.00`, tests passed |
| 8 | `uv run python -m src.generator` | prompt markup without weights | `[Source 1] ... lora.md (4695-5100)`, `[Prompt Template Structure Verified]` |

Watch: Step 6 must score Rank #1 **above 1.5** on the LoRA query; Step 5
must assert `sample[start:end] == chunk.text` for every chunk ( grading
fails on a single broken offset).

### Phase 2 — Lint (Chapter V.1)

```bash
make lint         # flake8 src api + mypy with subject flags
make lint-strict  # flake8 + mypy --strict (recommended extra)
```

Watch: must print `Success: no issues found in 15 source files` with zero
warnings. Lint scope is `src api` on purpose: the vendored corpus under
`data/raw/` contains duplicate module names that break bare `mypy .`.
Line length limit is 100 (`setup.cfg`).

### Phase 3 — Degenerate inputs (CLI must never traceback)

```bash
uv run python -m src index --max_chunk_size 5000   # stderr error, exit 0
uv run python -m src search "" --k 2               # prints []
uv run python -m src search "test" --k 0           # prints []
uv run python -m src search "test" --k 2 --index_directory /nope  # [] + stderr
```

Watch: every bad input prints `Error: ...` to **stderr** and a safe empty
payload (`[]` / `{}`) to stdout, exit code stays 0. The reviewer tests
empty queries, `k=0`, missing files and malformed JSON — any unhandled
traceback counts as non-functional.

### Phase 4 — Full pipeline (mirrors the exam scripts)

```bash
# 1. Index once (~10 s lexical; limit 5 min)
uv run python -m src index --max_chunk_size 2000
# -> "Ingestion complete! Indexed 28837 chunks under data/processed/"
# Re-run must print "Index up-to-date! ..." (Bonus 3: nothing changed)

# 2. Single search (spot-check grounding)
uv run python -m src search "What HTTP endpoint is used to dynamically load a LoRA adapter?" --k 1
# -> [{"file_path": "data/raw/vllm-0.10.1/docs/features/lora.md",
#      "first_character_index": 4695, "last_character_index": 6100}]

# 3. Batch search, one output per dataset (scoping avoids overwrites!)
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 --save_directory data/output/search_results/UnansweredQuestions
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json \
  --k 10 --save_directory data/output/search_results/UnansweredQuestions
# -> "Saved student_search_results to .../UnansweredQuestions/dataset_*_public.json"

# 4. Self-check with our own evaluator (same math as the moulinette)
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json

# 5. Official scoring (Linux binary; on Windows run it from WSL, see note)
./moulinette/moulinette evaluate_student_search_results \
  data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10 --max_context_length 2000

# 6. Answers (downloads Qwen3-0.6B once; ~100 s/question on CPU)
uv run python -m src answer "What HTTP endpoint loads a LoRA adapter?" --k 2
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

WSL note: the moulinette is a Linux ELF binary, so on a Windows station
run `wsl -d Ubuntu bash -c "cd /mnt/d/Project/4_42Projects/RAG_42Ber && ..."`
with the same command line. Verified results:

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Verdict |
|---|---|---|---|---|---|
| docs (100 q) | 0.610 | 0.750 | **0.810** | 0.860 | PASS >= 80% |
| code (99 q) | 0.333 | 0.495 | **0.545** | 0.606 | PASS >= 50% |

Watch: `Student data is valid: True` must appear first — otherwise the
JSON schema is wrong and scores are meaningless. Our `evaluate` must agree
with the moulinette to 3 decimals; if they diverge, the IoU/path logic
regressed.

### Phase 5 — Output files inventory (where everything lands)

All paths are CLI arguments, never hardcoded; these are just the defaults:

```text
data/processed/
  chunks.json                 # 28,837 chunks (26 MB) — full span + text store
  meta.json                   # {"num_chunks": 28837, "created_at": ...}
  hashes.json                 # SHA-256 per file (Bonus 3 incremental state)
  query_cache.json            # cached query results (Bonus 4)
  bm25_index/                 # BM25 matrices + vocabs (bulk of the index)
    data.csc.index.npy, indices.csc.index.npy, indptr.csc.index.npy
    params.index.json, vocab.index.json, vocab.tokenizer.json
  dense_index/                # only with `index --dense True` (Bonus 1)
    embeddings.npy, order.json

data/output/search_results/<DatasetScope>/      # StudentSearchResults JSON
  UnansweredQuestions/dataset_docs_public.json  # 100 q x 10 sources
  UnansweredQuestions/dataset_code_public.json  # 99 q x 10 sources

data/output/search_results_and_answer/<DatasetScope>/  # ...AndAnswer JSON
  UnansweredQuestions/dataset_docs_public.json  # same + "answer" per question
```

Validate any output file in one line (must print `over2000: 0`):

```bash
uv run python -c "import json,glob; r=[]; \
[r.extend(json.load(open(f,encoding='utf-8'))['search_results']) \
for f in glob.glob('data/output/search_results/*/*.json')]; \
L=[s['last_character_index']-s['first_character_index'] \
for e in r for s in e['retrieved_sources']]; \
print('total_sources:',len(L),'max_len:',max(L), \
'over2000:',sum(1 for x in L if x>2000))"
# -> total_sources: 1990 max_len: 2000 over2000: 0
```

Watch: `<DatasetScope>` (`UnansweredQuestions` vs `AnsweredQuestions`) is
mandatory — both public datasets share filenames, so an unscoped
`--save_directory` silently overwrites the previous run. Every
`file_path` must read exactly `data/raw/vllm-0.10.1/...` (logical path
equals disk path relative to project root — if you re-ingest a
differently nested corpus and paths come out doubled, recall
drops to ~0).

### Phase 6 — Bonus verification (each must be *demonstrated*, not described)

```bash
# Bonus 1+2: dense + hybrid (MiniLM download on first use)
uv run python -m src index --max_chunk_size 2000 --dense True
# -> data/processed/dense_index/embeddings.npy appears
# Hybrid RRF (1/(60+rank)) lives in src/dense.py::rrf_fuse, wired into
# search via the hybrid path with BM25 fallback when dense is absent.

# Bonus 3: incremental — second identical run is a no-op
uv run python -m src index --max_chunk_size 2000
# -> "Index up-to-date! 28837 chunks ..."
uv run python -c "from src.hash import compute_hash, diff_hashes; \
print('Hash OK:', compute_hash('test'))"

# Bonus 4: cache — repeat queries skip retrieval
uv run python -c "from src.cache import QueryCache; \
c = QueryCache(); print('Cache ready!')"
# data/processed/query_cache.json grows; it self-invalidates when
# meta.json created_at changes (re-index).

# Bonus 5: HTTP API
make api  # :8000, POST /search {"query","k"} and POST /answer
curl -X POST localhost:8000/search -H "Content-Type: application/json" \
  -d '{"query": "How to configure LoRA?", "k": 2}'
# -> 200 with MinimalSource list; retrieval failure -> HTTP 500 JSON error
```

Watch: bonuses are graded only after the mandatory part fully validates —
never demo a bonus on a broken pipeline.

### Phase 7 — Defense-day checklist (reviewer perspective)

1. `make install && make lint` green before anything else.
2. `make run`, then the Phase 4 sequence; confirm both moulinette
   `Recall@5` bars and `Student data is valid: True`.
3. Probe robustness live: empty query, `k=0`, unknown paths — expect `[]`
   + stderr, never a traceback.
4. Ask for a recode (Chapter X.1): the code is deliberately boring —
   each module is one idea in ~100 lines, so a small behavior change
   (e.g. different overlap, new CLI flag, extra RRF constant) lands in
   minutes; `make lint` re-verifies.
5. Know the numbers: 1952 files -> 28,837 chunks, ~10 s index,
   ~140 q/s retrieval, docs 0.81 / code 0.545, all spans <= 2000 chars.

## Resources

- vLLM documentation (indexed corpus itself, `data/raw/vllm-0.10.1/docs/`)
- BM25 (Robertson & Zaragoza) + `bm25s` library docs
- Sentence-Transformers `all-MiniLM-L6-v2` model card
- Qwen3-0.6B model card (Qwen)
- FastAPI documentation; Python Fire documentation
- 42 subject `docs/en.subject.pdf` + `docs/step_by_step_specification.md`

### AI usage

AI assistance (code-oriented LLM agent) was used for: scaffolding module
skeletons from the step-by-step specification, drafting docstrings and the
README structure, and diagnosing errors (path prefix duplication, lint
scope). All retrieval thresholds were verified by executing the real
pipeline (index -> search_dataset -> evaluate); every diagnostic and metric
quoted above comes from actual runs. Final code was reviewed, trimmed to
minimal size, and kept under human control per the 42 AI instructions.
