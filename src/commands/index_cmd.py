"""Build chunk index."""

import json
from pathlib import Path
from typing import List

from tqdm import tqdm

from src.chunking import Chunk, chunk_file
from src.hash import diff_hashes, hashes_for
from src.index import KnowledgeBase
from src.ingest import collect_files
from src.retrieval import err

from src.dense import DenseIndex


def run_index(
    repo_path: str = "data/raw/vllm-0.10.1",
    save_directory: str = "data/processed",
    max_chunk_size: int = 2000,
    dense: bool = False,
) -> None:
    """Chunk corpus and persist index."""
    if not isinstance(max_chunk_size, int) or not 0 < max_chunk_size <= 2000:
        err(f"invalid max_chunk_size={max_chunk_size!r}: need 0 < v <= 2000")
        return

    if not Path(repo_path).is_dir():
        err(f"repo path not found: '{repo_path}'")
        return

    try:
        save_root = Path(save_directory)
        save_root.mkdir(parents=True, exist_ok=True)

        try:
            old_hashes = json.loads(
                (save_root / "hashes.json").read_text("utf-8")
            )
        except (OSError, ValueError):
            old_hashes = {}

        files = collect_files(repo_path, relative_to=".", file_type="all")
        cur_hashes = hashes_for(files)
        changed, deleted = diff_hashes(old_hashes, cur_hashes)

        # for idx, (path, file_hash) in enumerate(
        #   list(cur_hashes.items())[:10], start=1
        # ):
        #     print(f"[{idx}] File: {path}")
        #     print(f"Hash: {file_hash}")
        #     print("-" * 50)

        # for idx, (path, cont) in enumerate(files[:10], start=1):
        #     print(f"[{idx}] File: {path}")
        #     print(f"Content:\n {cont[:100]}")
        #     print("-" * 50)

        try:
            raw = json.loads((save_root / "chunks.json").read_text("utf-8"))
            old_chunks = [Chunk(**r) for r in raw if isinstance(r, dict)]
        except (OSError, ValueError, TypeError):
            old_chunks = []

        if old_chunks and (save_root / "bm25_index").is_dir():
            if not changed and not deleted:
                print(
                    f"Index up-to-date! {len(old_chunks)}"
                    f"chunks under {save_directory}/"
                )
                return

        keep = []
        for c in old_chunks:
            if c.file_path not in changed:
                if c.file_path not in deleted:
                    keep.append(c)

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
                err(f"skip '{path}': {exc}")

        # for chunk in chunks[-5:]:
        #     print(f"\n {chunk}")
        #     print("-" * 40)

        kb = KnowledgeBase(chunks)
        tqdm.write(f"Building BM25 index for {len(chunks)} chunks...")
        kb.build(show_progress=True)
        kb.save(save_directory, show_progress=True)

        (save_root / "hashes.json").write_text(
            json.dumps(cur_hashes, indent=2), encoding="utf-8")
        if dense:
            try:
                tqdm.write(f"Building dense index for {len(chunks)} chunks...")
                idx = DenseIndex(chunks)
                idx.build(show_progress=True)
                idx.save(str(save_root / "dense_index"))
            except Exception as exc:
                err(f"dense index skipped: {exc}")
        print(f"Ingestion complete! Indexed {len(chunks)}"
              f"chunks under {save_directory}/")
    except Exception as exc:
        err(f"indexing failed: {exc}")
