"""Fire CLI with 6 mandatory commands (Step 9) plus bonus wiring."""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from src.cache import QueryCache
from src.chunking import Chunk, chunk_file
from src.evaluator import KS, recall_at_k
from src.hash import diff_hashes, hashes_for
from src.index import KnowledgeBase
from src.ingest import collect_files, read_file_safely
from src.models import MinimalAnswer, MinimalSearchResults, MinimalSource
from src.models import StudentSearchResults, StudentSearchResultsAndAnswer


def _err(msg: str) -> None:
    """Print error to stderr without traceback."""
    print(f"Error: {msg}", file=sys.stderr)


def _load_kb(index_directory: str) -> Optional[KnowledgeBase]:
    """Load KnowledgeBase or report error gracefully."""
    try:
        return KnowledgeBase.load(index_directory)
    except Exception as exc:
        _err(f"cannot load index from '{index_directory}': {exc}")
        return None


def _fingerprint(index_directory: str) -> str:
    """Read index fingerprint for cache invalidation."""
    try:
        meta = json.loads((Path(index_directory) / "meta.json").read_text("utf-8"))
        fp = meta.get("created_at", "")
        return str(fp)
    except (OSError, ValueError):
        return ""


def _real_root(repo_path: str) -> str:
    """Resolve doubled extraction nesting (vllm-0.10.1/vllm-0.10.1).

    Args:
        repo_path: Configured corpus root.

    Returns:
        Inner directory when extraction nesting is detected, else repo_path.
    """
    inner = str(Path(repo_path) / Path(repo_path).name)
    if Path(inner).is_dir():
        return inner
    return repo_path


def _disk_path(file_path: str, repo_path: str = "data/raw/vllm-0.10.1") -> str:
    """Map a logical file_path back to its on-disk location.

    Args:
        file_path: Logical path as stored in the index.
        repo_path: Configured corpus root.

    Returns:
        Existing filesystem path or "" when not found.
    """
    if Path(file_path).is_file():
        return file_path
    prefix = repo_path.replace("\\", "/").rstrip("/")
    if file_path.startswith(prefix + "/"):
        rel = file_path[len(prefix) + 1:]
        cand = str(Path(_real_root(repo_path)) / rel).replace("\\", "/")
        if Path(cand).is_file():
            return cand
    return ""


def _chunk_to_source(ch: Chunk) -> MinimalSource:
    """Convert Chunk to MinimalSource output model."""
    return MinimalSource(
        file_path=ch.file_path,
        first_character_index=ch.first_character_index,
        last_character_index=ch.last_character_index,
    )


def _search_cached(
    kb: KnowledgeBase, query: str, k: int, file_type: str, cache: QueryCache,
    hybrid: bool = False,
) -> List[MinimalSource]:
    """Search with cache lookup, storing results on miss."""
    hit = cache.get(query, k)
    if hit is not None:
        return hit[: max(k, 0)]
    pairs = _hybrid_search(kb, query, k, file_type) if hybrid else kb.search_bm25(
        query, k=k, file_type=file_type
    )
    sources = [_chunk_to_source(ch) for ch, _ in pairs]
    cache.put(query, k, sources)
    return sources


def _hybrid_search(
    kb: KnowledgeBase, query: str, k: int, file_type: str
) -> List[tuple[Chunk, float]]:
    """Fuse BM25 and dense rankings with RRF, fallback to BM25."""
    bm25_pairs = kb.search_bm25(query, k=k, file_type=file_type)
    try:
        from src.dense import DenseIndex, rrf_fuse

        dense = DenseIndex.load(str(Path(_kb_dir[0]) / "dense_index"))
        dense.chunks = kb.chunks
        dense_pairs = dense.search_dense(query, k=k, file_type=file_type)
        if not dense_pairs:
            return bm25_pairs

        def key_of(ch: Chunk) -> str:
            """Build dedup key for a chunk."""
            return f"{ch.file_path}:{ch.first_character_index}"

        by_key: Dict[str, tuple[Chunk, float]] = {}
        for ch, sc in bm25_pairs + dense_pairs:
            by_key.setdefault(key_of(ch), (ch, sc))
        fused = rrf_fuse(
            [[key_of(ch) for ch, _ in bm25_pairs],
             [key_of(ch) for ch, _ in dense_pairs]], k=k,
        )
        return [(by_key[key][0], score) for key, score in fused if key in by_key]
    except Exception:
        return bm25_pairs


_kb_dir: List[str] = ["data/processed"]


def _resolve_chunks(
    sources: List[MinimalSource], index_directory: str
) -> List[Chunk]:
    """Map retrieved sources back to chunk texts via file coordinates."""
    try:
        kb = KnowledgeBase.load(index_directory)
        by_span = {(c.file_path, c.first_character_index): c for c in kb.chunks}
    except Exception:
        by_span = {}
    chunks: List[Chunk] = []
    for src in sources:
        hit = by_span.get((src.file_path, src.first_character_index))
        if hit is not None and hit.last_character_index == src.last_character_index:
            chunks.append(hit)
            continue
        disk = _disk_path(src.file_path)
        text = read_file_safely(disk) if disk else ""
        span = text[src.first_character_index:src.last_character_index]
        chunks.append(Chunk(src.file_path, src.first_character_index,
                            src.last_character_index, span))
    return chunks


class CLI:
    """Entry point for `python -m src <command>` (Chapter VI.6)."""

    def index(
        self,
        repo_path: str = "data/raw/vllm-0.10.1",
        save_directory: str = "data/processed",
        max_chunk_size: int = 2000,
        dense: bool = False,
    ) -> None:
        """Chunk corpus and persist index (incremental via SHA-256).

        Args:
            repo_path: Corpus root to ingest.
            save_directory: Index output directory.
            max_chunk_size: Max chunk length (0 < v <= 2000).
            dense: Also build MiniLM dense index (Bonus 1, slow on CPU).
        """
        if not isinstance(max_chunk_size, int) or not 0 < max_chunk_size <= 2000:
            _err(f"invalid max_chunk_size={max_chunk_size!r}: need 0 < v <= 2000")
            return
        if not Path(repo_path).is_dir():
            _err(f"repo path not found: '{repo_path}'")
            return
        try:
            save_root = Path(save_directory)
            save_root.mkdir(parents=True, exist_ok=True)
            old_hashes: Dict[str, str] = {}
            try:
                old_hashes = json.loads(
                    (save_root / "hashes.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                old_hashes = {}
            root = _real_root(repo_path)
            files = collect_files(root, relative_to=root, file_type="all")
            prefix = repo_path.replace("\\", "/").rstrip("/")
            files = [(f"{prefix}/{p}" if not p.startswith(prefix + "/") else p, t)
                     for p, t in files]
            cur_hashes = hashes_for(files)
            changed, deleted = diff_hashes(old_hashes, cur_hashes)
            old_chunks: List[Chunk] = []
            try:
                raw = json.loads((save_root / "chunks.json").read_text("utf-8"))
                old_chunks = [Chunk(**r) for r in raw if isinstance(r, dict)]
            except (OSError, ValueError, TypeError):
                old_chunks = []
            bm25_ok = (save_root / "bm25_index").is_dir()
            if old_chunks and bm25_ok and not changed and not deleted:
                print(f"Index up-to-date! {len(old_chunks)} chunks "
                      f"under {save_directory}/")
                return
            keep = [c for c in old_chunks
                    if c.file_path not in changed and c.file_path not in deleted]
            by_path = {p: t for p, t in files}
            chunks: List[Chunk] = list(keep)
            targets = sorted(changed) if old_chunks else sorted(by_path)
            for path in tqdm(targets, desc="Chunking", unit="file"):
                text = by_path.get(path)
                if text is None:
                    continue
                try:
                    chunks.extend(chunk_file(path, text, max_chunk_size))
                except Exception as exc:
                    _err(f"skip '{path}': {exc}")
            kb = KnowledgeBase(chunks)
            tqdm.write("Building BM25 index...")
            kb.build()
            kb.save(save_directory)
            (save_root / "hashes.json").write_text(
                json.dumps(cur_hashes, indent=2), encoding="utf-8")
            if dense:
                try:
                    from src.dense import DenseIndex

                    dense_idx = DenseIndex(chunks)
                    dense_idx.build()
                    dense_idx.save(str(save_root / "dense_index"))
                except Exception as exc:
                    _err(f"dense index skipped: {exc}")
            print(f"Ingestion complete! Indexed {len(chunks)} chunks "
                  f"under {save_directory}/")
        except Exception as exc:
            _err(f"indexing failed: {exc}")

    def search(
        self,
        query: str = "",
        k: int = 10,
        index_directory: str = "data/processed",
        file_type: str = "all",
    ) -> None:
        """Return top-k sources for a single query as JSON list.

        Args:
            query: User question string.
            k: Number of results.
            index_directory: Index directory.
            file_type: "all", "docs" or "code".
        """
        try:
            if not query or not query.strip() or k <= 0:
                print("[]")
                return
            if file_type not in ("all", "docs", "code"):
                _err(f"invalid file_type={file_type!r}")
                print("[]")
                return
            kb = _load_kb(index_directory)
            if kb is None:
                print("[]")
                return
            _kb_dir[0] = index_directory
            cache = QueryCache(fingerprint=_fingerprint(index_directory))
            sources = _search_cached(kb, query, k, file_type, cache)
            print(json.dumps([s.model_dump() for s in sources]))
        except Exception as exc:
            _err(f"search failed: {exc}")
            print("[]")

    def search_dataset(
        self,
        dataset_path: str = "",
        k: int = 10,
        save_directory: str = "data/output/search_results",
        index_directory: str = "data/processed",
        file_type: str = "all",
    ) -> None:
        """Run search over a dataset and write StudentSearchResults JSON.

        Args:
            dataset_path: Input dataset JSON path.
            k: Number of results per question.
            save_directory: Output directory.
            index_directory: Index directory.
            file_type: "all", "docs" or "code".
        """
        try:
            kb = _load_kb(index_directory)
            if kb is None:
                return
            try:
                data = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                _err(f"cannot read dataset '{dataset_path}': {exc}")
                return
            questions = data.get("rag_questions", []) if isinstance(data, dict) else []
            _kb_dir[0] = index_directory
            cache = QueryCache(fingerprint=_fingerprint(index_directory))
            results = []
            for q in tqdm(questions, desc="Searching", unit="q"):
                try:
                    qid = str(q.get("question_id", ""))
                    text = str(q.get("question", ""))
                    srcs = _search_cached(kb, text, k, file_type, cache) if text else []
                    results.append(MinimalSearchResults(
                        question_id=qid, question=text,
                        retrieved_sources=srcs).model_dump())
                except Exception as exc:
                    _err(f"skip question: {exc}")
            out = StudentSearchResults(
                search_results=[MinimalSearchResults(**r) for r in results], k=k)
            save_root = Path(save_directory)
            save_root.mkdir(parents=True, exist_ok=True)
            dest = save_root / Path(dataset_path).name
            dest.write_text(out.model_dump_json(indent=2), encoding="utf-8")
            print(f"Saved student_search_results to {dest.as_posix()}")
        except Exception as exc:
            _err(f"search_dataset failed: {exc}")

    def answer(
        self,
        query: str = "",
        k: int = 5,
        index_directory: str = "data/processed",
    ) -> None:
        """Answer a single query using retrieved context.

        Args:
            query: User question string.
            k: Number of context chunks.
            index_directory: Index directory.
        """
        try:
            if not query or not query.strip() or k <= 0:
                print(json.dumps({"question": query, "answer": "", "sources": []}))
                return
            kb = _load_kb(index_directory)
            if kb is None:
                return
            pairs = kb.search_bm25(query, k=k)
            chunks = [ch for ch, _ in pairs]
            try:
                from src.generator import get_generator

                text = get_generator().generate(query, chunks)
            except Exception as exc:
                _err(f"generation failed: {exc}")
                text = ""
            ans = MinimalAnswer(question_id="single-query", question=query,
                                retrieved_sources=[_chunk_to_source(c) for c in chunks],
                                answer=text)
            print(ans.model_dump_json(indent=2))
        except Exception as exc:
            _err(f"answer failed: {exc}")

    def answer_dataset(
        self,
        student_search_results_path: str = "",
        save_directory: str = "data/output/search_results_and_answer",
        index_directory: str = "data/processed",
    ) -> None:
        """Generate answers for search results, writing ...AndAnswer JSON.

        Args:
            student_search_results_path: Input search results JSON.
            save_directory: Output directory.
            index_directory: Index directory for chunk texts.
        """
        try:
            try:
                data = json.loads(
                    Path(student_search_results_path).read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                _err(f"cannot read '{student_search_results_path}': {exc}")
                return
            entries = data.get("search_results", []) if isinstance(data, dict) else []
            kk = int(data.get("k", 5)) if isinstance(data, dict) else 5
            from src.generator import get_generator

            gen = get_generator()
            answers = []
            print(f"Loaded {len(entries)} questions ...")
            for e in tqdm(entries, desc="Answering", unit="q"):
                try:
                    srcs = [MinimalSource(**s) for s in e.get("retrieved_sources", [])]
                    chunks = _resolve_chunks(srcs, index_directory)
                    try:
                        text = gen.generate(str(e.get("question", "")), chunks)
                    except Exception as exc:
                        _err(f"generation failed: {exc}")
                        text = ""
                    answers.append(MinimalAnswer(
                        question_id=str(e.get("question_id", "")),
                        question=str(e.get("question", "")),
                        retrieved_sources=srcs, answer=text))
                except Exception as exc:
                    _err(f"skip entry: {exc}")
            out = StudentSearchResultsAndAnswer(search_results=answers, k=kk)
            save_root = Path(save_directory)
            save_root.mkdir(parents=True, exist_ok=True)
            dest = save_root / Path(student_search_results_path).name
            dest.write_text(out.model_dump_json(indent=2), encoding="utf-8")
            print(f"Processed {len(answers)} of {len(entries)} questions")
            print(f"Saved student_search_results_and_answer to {dest.as_posix()}")
        except Exception as exc:
            _err(f"answer_dataset failed: {exc}")

    def evaluate(
        self, student_search_results_path: str = "", dataset_path: str = ""
    ) -> None:
        """Report Recall@k against a ground-truth dataset.

        Args:
            student_search_results_path: Student results JSON.
            dataset_path: Ground truth dataset JSON.
        """
        try:
            try:
                student = json.loads(
                    Path(student_search_results_path).read_text(encoding="utf-8"))
                truth = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                _err(f"cannot read input: {exc}")
                return
            truths = {q.get("question_id"): q.get("sources", [])
                      for q in truth.get("rag_questions", [])}
            entries = student.get("search_results", [])
            per_k: Dict[int, List[float]] = {k: [] for k in KS}
            for e in entries:
                tlist = [MinimalSource(**s) for s in truths.get(
                    e.get("question_id"), []) if isinstance(s, dict)]
                rlist = [MinimalSource(**s) for s in e.get("retrieved_sources", [])
                         if isinstance(s, dict)]
                for k in KS:
                    per_k[k].append(recall_at_k(rlist, tlist, k))
            print("Evaluation Results\n" + "=" * 40)
            for k in KS:
                vals = per_k[k]
                avg = sum(vals) / len(vals) if vals else 0.0
                mark = " (PASS >= 80%)" if k == 5 and avg >= 0.8 else ""
                print(f"Recall@{k}: {avg:.3f}{mark}")
        except Exception as exc:
            _err(f"evaluate failed: {exc}")
