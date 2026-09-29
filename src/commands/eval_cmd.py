"""Evaluate command."""

import json
from pathlib import Path
from typing import Dict, List

from src.evaluator import KS, recall_at_k
from src.models import MinimalSource
from src.retrieval import err


def run_evaluate(student_path: str, dataset_path: str) -> None:
    """Print Recall@k against ground truth."""
    try:
        try:
            student = json.loads(Path(student_path).read_text(encoding="utf-8"))
            truth = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            err(f"cannot read input: {exc}")
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
        err(f"evaluate failed: {exc}")
