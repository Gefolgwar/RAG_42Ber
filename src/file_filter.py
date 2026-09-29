"""One rule for docs/code file kinds."""

import os

DOC_EXTS = frozenset({".md", ".markdown", ".rst", ".txt"})
CODE_EXTS = frozenset({".py"})


def matches(path: str, file_type: str) -> bool:
    """True when path belongs to file_type (all/docs/code)."""
    ext = os.path.splitext(path)[1].lower()
    if file_type == "docs":
        return ext in DOC_EXTS
    if file_type == "code":
        return ext in CODE_EXTS
    return ext in DOC_EXTS or ext in CODE_EXTS
