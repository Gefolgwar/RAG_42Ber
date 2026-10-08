"""Corpus scanning and safe file reading (Step 4: Ingestion).

Recursively discovers documentation and Python sources under ``data/raw/``,
filters them by extension and directory blocklist, and reads contents as
UTF-8 with error replacement so a single binary artefact never crashes
indexing. All returned ``file_path`` values are relative to the repository
root with forward slashes, as required for verbatim grading.
"""

import os
from collections.abc import Iterator
from typing import List, Tuple

from src.file_filter import matches

IGNORED_DIRS = frozenset(
    {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "build",
        "dist",
        ".github",
        ".buildkite",
        "node_modules",
    }
)


def iter_files(root: str, file_type: str = "all") -> Iterator[str]:
    """Yield filesystem paths of corpus files under root.

    Args:
        root: Directory to scan recursively.
        file_type: Filter category: ``"all"``, ``"docs"`` or ``"code"``.

    Yields:
        Filesystem paths (joined with ``root``) matching the whitelist
        and not located inside a blocklisted directory.
    """
    try:
        walk = os.walk(root)
    except OSError:
        return
    for dirpath, dirnames, filenames in walk:
        try:
            dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
            for name in sorted(filenames):
                if matches(name, file_type):
                    yield os.path.join(dirpath, name)
        except OSError:
            continue


def read_file_safely(path: str) -> str:
    """Read a text file without ever raising on decode errors.

    Args:
        path: Filesystem path to read.

    Returns:
        File content decoded as UTF-8 with undecodable bytes replaced.
        Returns an empty string when the file cannot be opened.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            content: str = handle.read()
        return content
    except OSError:
        return ""


def collect_files(
    root: str,
    relative_to: str = ".",
    file_type: str = "all",
) -> List[Tuple[str, str]]:
    """Collect (relative_path, text) pairs for every corpus file.

    Args:
        root: Directory to scan recursively.
        relative_to: Base directory for relative ``file_path`` values.
        file_type: Filter category: ``"all"``, ``"docs"`` or ``"code"``.

    Returns:
        List of tuples where the first element is the path relative to
        ``relative_to`` with forward slashes and the second element is
        the safely decoded file content.
    """
    results: List[Tuple[str, str]] = []

    for fs_path in iter_files(root, file_type=file_type):
        try:
            rel: str = os.path.relpath(fs_path, relative_to)
        except ValueError:
            continue

        rel = rel.replace(os.sep, "/")
        text: str = read_file_safely(fs_path)
        results.append((rel, text))

    return results


if __name__ == "__main__":
    from tqdm import tqdm

    print("=== [STEP 4 DIAGNOSTIC] Testing File Ingestion ===")
    corpus_root: str = "data/raw"
    try:
        file_list: List[str] = list(
            tqdm(
                iter_files(corpus_root),
                desc="Collecting corpus files",
                unit="file",
            )
        )
    except OSError as exc:
        print(f"[ERROR] Cannot scan corpus root '{corpus_root}': {exc}")
        raise SystemExit(1)
    print(f"[Summary] Total files discovered: {len(file_list)}")
    print("[First 5 files]:")
    for fs_path in file_list[:5]:
        rel_path: str = os.path.relpath(fs_path, ".").replace(os.sep, "/")
        content: str = read_file_safely(fs_path)
        line_count: int = content.count("\n") + (1 if content else 0)
        print(
            f"  - Path: {rel_path} | "
            f"Length: {len(content)} chars | Lines: {line_count}"
        )
