"""SHA-256 file hashing for incremental indexing (Bonus 3)."""

import hashlib
from typing import Dict, List, Set, Tuple


def compute_hash(text: str) -> str:
    """Compute hex SHA-256 of file content.

    Args:
        text: File content string.

    Returns:
        64-char hex digest.
    """
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def diff_hashes(
    old_hashes: Dict[str, str],
    current_hashes: Dict[str, str]
) -> Tuple[Set[str], Set[str]]:
    """Diff two path->hash maps.

    Args:
        old_hashes: Hashes from previous indexing run.
        current_hashes: Hashes from current scan.

    Returns:
        Tuple (changed_or_new, deleted) as sets of relative paths.
    """
    old_keys = set(old_hashes)
    cur_keys = set(current_hashes)
    changed = {p for p in cur_keys if current_hashes[p] != old_hashes.get(p)}
    deleted = old_keys - cur_keys
    return changed, deleted


def hashes_for(
    files: List[Tuple[str, str]],
    file_hashes: Dict[str, str] | None = None
) -> Dict[str, str]:
    """Build path->hash map for collected files.

    Args:
        files: (relative_path, text) pairs.
        file_hashes: Optional dict to fill and return.

    Returns:
        Dict mapping relative path to SHA-256 hex.
    """
    out: Dict[str, str] = file_hashes if file_hashes is not None else {}
    for path, text in files:
        out[path] = compute_hash(text)
    return out
