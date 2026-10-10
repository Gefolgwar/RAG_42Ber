"""Search commands."""

import json
from pathlib import Path

from tqdm import tqdm

from src.cache import QueryCache
from src.models import MinimalSearchResults, StudentSearchResults
from src.retrieval import err, fingerprint, load_kb, search_cached


def run_search(
    query: str, k: int, index_directory: str, file_type: str, hybrid: bool = False
) -> None:
    """Print top-k sources as JSON list."""
    try:
        if not query or not query.strip() or k <= 0:
            print("[]")
            return
        if file_type not in ("all", "docs", "code"):
            err(f"invalid file_type={file_type!r}")
            print("[]")
            return
        # Load the KnowledgeBase (BM25 index + chunks)
        kb = load_kb(index_directory)
        if kb is None:
            print("[]")
            return
        # fingerprint(...) read 'created_at' from data/processed/meta.json.
        cache = QueryCache(fingerprint=fingerprint(index_directory))
        # Search with cache, fallback to BM25 or hybrid search
        sources = search_cached(kb, query, k, file_type, cache, index_directory, hybrid)
        print(json.dumps([s.model_dump() for s in sources]))
    except Exception as exc:
        err(f"search failed: {exc}")
        print("[]")


def run_search_dataset(
    dataset_path: str,
    k: int,
    save_directory: str,
    index_directory: str,
    file_type: str,
    hybrid: bool = False,
) -> None:
    """Search dataset and write results JSON."""
    try:
        kb = load_kb(index_directory)
        if kb is None:
            return
        try:
            data = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            err(f"cannot read dataset '{dataset_path}': {exc}")
            return
        questions = data.get("rag_questions", []) if isinstance(data, dict) else []
        cache = QueryCache(fingerprint=fingerprint(index_directory))
        results = []
        for q in tqdm(questions, desc="Searching", unit="q"):
            try:
                qid = str(q.get("question_id", ""))
                text = str(q.get("question", ""))
                if text:
                    srcs = search_cached(
                        kb, text, k, file_type, cache, index_directory, hybrid
                    )
                else:
                    srcs = []
                results.append(MinimalSearchResults(
                    question_id=qid, question=text,
                    retrieved_sources=srcs).model_dump())
            except Exception as exc:
                err(f"skip question: {exc}")
        # region [explanation and example] How does unpacking **r work in StudentSearchResults
        # 1. INPUT EXAMPLE (what is currently in the results list):
        # results = [
        #     {
        #         "question_id": "q1",
        #         "question": "How to load LoRA?",
        #         "retrieved_sources": [
        #             {"file_path": "docs/lora.md", "first_character_index": 10, "last_character_index": 200}
        #         ]
        #     }
        # ]
        #
        # 2. UNPACKING MECHANISM (**r):
        # For each dictionary r, the expression MinimalSearchResults(**r) translates to a direct call:
        # MinimalSearchResults(
        #     question_id="q1",
        #     question="How to load LoRA?",
        #     retrieved_sources=[...]
        # )
        #
        # 3. OUTPUT EXAMPLE (what will appear in the out variable):
        # out = StudentSearchResults(
        #     search_results=[
        #         MinimalSearchResults(question_id='q1', question='How to load LoRA?', ...)
        #     ],
        #     k=10
        # )
        # endregion
        out = StudentSearchResults(
            search_results=[MinimalSearchResults(**r) for r in results], k=k)
        save_root = Path(save_directory)
        save_root.mkdir(parents=True, exist_ok=True)
        dest = save_root / Path(dataset_path).name
        dest.write_text(out.model_dump_json(indent=2), encoding="utf-8")
        # as_posix() to get the path in correct format for both Windows and Unix-like systems
        print(f"Saved student_search_results to {dest.as_posix()}") 
    except Exception as exc:
        err(f"search_dataset failed: {exc}")
