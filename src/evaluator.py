"""Recall@k evaluation with IoU >= 0.05 (Step 7).

A truth source counts as found only on exact file_path match
plus character overlap over ground truth length >= threshold.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from src.models import MinimalSource

THRESHOLD = 0.05
KS: Tuple[int, ...] = (1, 3, 5, 10)


def overlap_ratio(retrieved: MinimalSource, truth: MinimalSource) -> float:
    """Compute intersection over ground truth length.

    Args:
        retrieved: System-retrieved source span.
        truth: Ground truth source span.

    Returns:
        Overlap in [0.0, 1.0]; 0.0 when files differ or truth is empty.
    """
    if retrieved.file_path != truth.file_path:
        return 0.0
    truth_len = truth.last_character_index - truth.first_character_index
    if truth_len <= 0:
        return 0.0
    inter = min(
        retrieved.last_character_index, truth.last_character_index
    ) - max(
        retrieved.first_character_index, truth.first_character_index
    )
    return max(0, inter) / truth_len


def recall_at_k(
    retrieved: List[MinimalSource],
    truths: List[MinimalSource],
    k: int,
    threshold: float = THRESHOLD,
) -> float:
    """Recall@k for a single question.

    Args:
        retrieved: Ranked retrieved sources.
        truths: Ground truth sources.
        k: Cutoff depth.
        threshold: Minimum IoU to count as hit.

    Returns:
        Share of truths with at least one valid hit in top-k.
    """
    if not truths or k <= 0:
        return 1.0 if not truths else 0.0
    top = retrieved[:k]
    hits = 0
    for truth in truths:
        if any(overlap_ratio(ret, truth) >= threshold for ret in top):
            hits += 1
    return hits / len(truths)


@dataclass
class EvalReport:
    """Averaged Recall@k over a dataset with formatted table."""

    recalls: Dict[int, float] = field(default_factory=dict)
    num_questions: int = 0

    def __str__(self) -> str:
        """Render Recall@1/3/5/10 table with PASS mark for Recall@5."""
        lines = [f"Questions evaluated: {self.num_questions}"]
        for k in KS:
            val = self.recalls.get(k, 0.0)
            mark = " (PASS >= 80%)" if k == 5 and val >= 0.8 else ""
            lines.append(f"Recall@{k}: {val:.3f}{mark}")
        return "\n".join(lines)


def evaluate(
    retrieved_per_q: Sequence[List[MinimalSource]],
    truths_per_q: Sequence[List[MinimalSource]],
    ks: Sequence[int] = KS,
) -> EvalReport:
    """Average Recall@k over all questions.

    Args:
        retrieved_per_q: Retrieved lists per question.
        truths_per_q: Ground truth lists per question.
        ks: Cutoff depths to report.

    Returns:
        EvalReport with mean recall per k.
    """
    n = len(truths_per_q)
    recalls: Dict[int, float] = {}
    for k in ks:
        total = sum(
            recall_at_k(list(retrieved_per_q[i]), list(truths_per_q[i]), k)
            for i in range(n)
        )
        recalls[int(k)] = total / n if n else 0.0
    return EvalReport(recalls=recalls, num_questions=n)
