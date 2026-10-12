"""Answer commands."""

import json
from pathlib import Path

from tqdm import tqdm

from src.models import (
    MinimalAnswer,
    MinimalSource,
    StudentSearchResultsAndAnswer
)
from src.retrieval import (
    chunk_to_source,
    err,
    hybrid_search,
    load_kb,
    resolve_chunks
)
from src.generator import get_generator


def run_answer(
        query: str,
        k: int,
        index_directory: str,
        hybrid: bool = False
) -> None:
    """Answer single query as JSON."""
    try:
        if not query or not query.strip() or k <= 0:
            print(json.dumps({"question": query, "answer": "", "sources": []}))
            return

        kb = load_kb(index_directory)

        if kb is None:
            return
        if hybrid:
            pairs = hybrid_search(kb, query, k, "all", index_directory)
        else:
            pairs = kb.search_bm25(query, k=k)
        chunks = [ch for ch, _ in pairs]
        try:
            text = get_generator().generate(query, chunks)
        except Exception as exc:
            err(f"generation failed: {exc}")
            text = ""
        ans = MinimalAnswer(
            question_id="single-query", question=query,
            retrieved_sources=[chunk_to_source(c) for c in chunks],
            answer=text
        )
        print(ans.model_dump_json(indent=2))
    except Exception as exc:
        err(f"answer failed: {exc}")


def run_answer_dataset(
    results_path: str, save_directory: str, index_directory: str
) -> None:
    """Answer dataset and write answers JSON."""
    try:
        try:
            data = json.loads(Path(results_path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            err(f"cannot read '{results_path}': {exc}")
            return
        entries = (
            data.get("search_results", []) if isinstance(data, dict) else []
        )
        kk = int(data.get("k", 5)) if isinstance(data, dict) else 5
        gen = get_generator()
        answers = []
        print(f"Loaded {len(entries)} questions ...")
        for e in tqdm(entries, desc="Answering", unit="q"):
            try:
                srcs = [
                    MinimalSource(**s) for s in e.get("retrieved_sources", [])
                ]
                chunks = resolve_chunks(srcs, index_directory)
                try:
                    text = gen.generate(str(e.get("question", "")), chunks)
                except Exception as exc:
                    err(f"generation failed: {exc}")
                    text = ""
                answers.append(MinimalAnswer(
                    question_id=str(e.get("question_id", "")),
                    question=str(e.get("question", "")),
                    retrieved_sources=srcs, answer=text))
            except Exception as exc:
                err(f"skip entry: {exc}")
        out = StudentSearchResultsAndAnswer(search_results=answers, k=kk)
        save_root = Path(save_directory)
        save_root.mkdir(parents=True, exist_ok=True)
        dest = save_root / Path(results_path).name
        dest.write_text(out.model_dump_json(indent=2), encoding="utf-8")
        print(f"Processed {len(answers)} of {len(entries)} questions")
        print(f"Saved student_search_results_and_answer to {dest.as_posix()}")
    except Exception as exc:
        err(f"answer_dataset failed: {exc}")
