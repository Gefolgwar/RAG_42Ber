"""Disk cache for repeated queries (Bonus 4)."""

import hashlib
import json
from pathlib import Path
from typing import List, Optional

from src.models import MinimalSource

CACHE_FILE = "data/processed/query_cache.json"


def cache_key(query: str, k: int, hybrid: bool = False) -> str:
    """Build SHA-256 key for normalized query plus k.

    Args:
        query: Raw user query.
        k: Requested depth.
        hybrid: Separate hybrid results from plain BM25 results.

    Returns:
        Hex digest key string.
    """
    norm = f"{query.strip().lower()}:{k}"
    if hybrid:
        norm += ":hybrid"
    return hashlib.sha256(norm.encode("utf-8", errors="replace")).hexdigest()


class QueryCache:
    """Persistent dict cache with index-fingerprint invalidation."""

    def __init__(self, path: str = CACHE_FILE, fingerprint: str = "") -> None:
        """Load cache, clearing it when the index fingerprint changed.

        Args:
            path: JSON cache file location.
            fingerprint: Current index fingerprint (e.g. meta created_at).
        """
        self.path = path
        self.fingerprint = fingerprint
        self._data: dict[str, object] = {"fingerprint": fingerprint, "items": {}}
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            if isinstance(raw, dict) and raw.get("fingerprint") == fingerprint:
                self._data = raw
        except OSError:
            pass
        except ValueError:
            pass

    def get(self, query: str, k: int, hybrid: bool = False) -> Optional[List[MinimalSource]]:
        """Return cached sources or None on miss.

        Args:
            query: User query string.
            k: Requested depth.
            hybrid: Whether the entry is for hybrid search.

        Returns:
            List of MinimalSource on hit, None on miss.
        """
        try:
            items = self._data.get("items")
            assert isinstance(items, dict)
            hit = items.get(cache_key(query, k, hybrid))
            if not isinstance(hit, list):
                return None
            return [MinimalSource(**s) for s in hit if isinstance(s, dict)]
        except (ValueError, TypeError, AssertionError):
            return None

    def put(
        self, query: str, k: int, sources: List[MinimalSource], hybrid: bool = False
    ) -> None:
        """Store sources and persist cache to disk.

        Args:
            query: User query string.
            k: Requested depth.
            sources: Sources to cache.
            hybrid: Whether the entry is for hybrid search.
        """
        try:
            items = self._data.get("items")
            if not isinstance(items, dict):
                items = {}
                self._data["items"] = items
            items[cache_key(query, k, hybrid)] = [s.model_dump() for s in sources]
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.path).write_text(json.dumps(self._data), encoding="utf-8")
        except OSError:
            pass
