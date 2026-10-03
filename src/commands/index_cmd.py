"""Build chunk index (Minimal version)."""

from pathlib import Path
from typing import List

from src.chunking import Chunk, chunk_file
from src.index import KnowledgeBase
from src.ingest import collect_files
from src.retrieval import err, real_root


def run_index(
    repo_path: str = "data/raw/vllm-0.10.1",
    save_directory: str = "data/processed",
    max_chunk_size: int = 2000,
    dense: bool = False,
) -> None:
    """Chunk corpus and persist index."""
    # 1. Валідація згідно з вимогами завдання
    if not isinstance(max_chunk_size, int) or not 0 < max_chunk_size <= 2000:
        err(f"invalid max_chunk_size={max_chunk_size!r}: need 0 < v <= 2000")
        return
    if not Path(repo_path).is_dir():
        err(f"repo path not found: '{repo_path}'")
        return

    try:
        # 2. Збір файлів та вирівнювання префікса шляхів
        root = real_root(repo_path)
        files = collect_files(root, relative_to=root, file_type="all")
        prefix = repo_path.replace("\\", "/").rstrip("/")
        files = [(p if p.startswith(prefix + "/") else f"{prefix}/{p}", t) for p, t in files]

        # 3. Чанкінг усіх файлів напряму у список
        chunks: List[Chunk] = []
        for path, text in files:
            try:
                chunks.extend(chunk_file(path, text, max_chunk_size))
            except Exception as exc:
                err(f"skip '{path}': {exc}")

        # 4. Побудова та збереження BM25 індексу
        kb = KnowledgeBase(chunks)
        kb.build()
        kb.save(save_directory)

        # 5. Опціональний Dense індекс (якщо передано --dense)
        if dense:
            try:
                from src.dense import DenseIndex
                idx = DenseIndex(chunks)
                idx.build()
                idx.save(str(Path(save_directory) / "dense_index"))
            except Exception as exc:
                err(f"dense index skipped: {exc}")

        print(f"Ingestion complete! Indexed {len(chunks)} chunks under {save_directory}/")

    except Exception as exc:
        err(f"indexing failed: {exc}")