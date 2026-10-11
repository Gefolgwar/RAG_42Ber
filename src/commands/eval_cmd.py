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

        # 50% for code, 80% for docs
        is_code = "code" in f"{dataset_path} {student_path}".lower()
        threshold = 0.5 if is_code else 0.8
        pct = 50 if is_code else 80

        print("Evaluation Results\n" + "=" * 40)
        for k in KS:
            vals = per_k[k]
            avg = sum(vals) / len(vals) if vals else 0.0
            mark = f" (PASS >= {pct}%)" if k == 5 and avg >= threshold else ""
            print(f"Recall@{k}: {avg:.3f}{mark}")
    except Exception as exc:
        err(f"evaluate failed: {exc}")