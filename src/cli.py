"""Thin CLI: delegates to commands, search via retrieval."""

from src.commands.answer_cmd import run_answer, run_answer_dataset
from src.commands.eval_cmd import run_evaluate
from src.commands.index_cmd import run_index
from src.commands.search_cmd import run_search, run_search_dataset


class CLI:
    """Entry point for `python -m src <command>`."""

    def index(
        self,
        repo_path: str = "data/raw/vllm-0.10.1",
        save_directory: str = "data/processed",
        max_chunk_size: int = 2000,
        dense: bool = False,
    ) -> None:
        """Chunk corpus and persist index."""
        run_index(repo_path, save_directory, max_chunk_size, dense)

    def search(
        self,
        query: str = "",
        k: int = 10,
        index_directory: str = "data/processed",
        file_type: str = "all",
        hybrid: bool = False,
    ) -> None:
        """Return top-k sources as JSON list."""
        run_search(query, k, index_directory, file_type, hybrid)

    def search_dataset(
        self,
        dataset_path: str = "",
        k: int = 10,
        save_directory: str = "data/output/search_results",
        index_directory: str = "data/processed",
        file_type: str = "all",
        hybrid: bool = False,
    ) -> None:
        """Search dataset and write results JSON."""
        run_search_dataset(
            dataset_path, k, save_directory, index_directory, file_type, hybrid
        )

    def answer(
        self,
        query: str = "",
        k: int = 5,
        index_directory: str = "data/processed",
        hybrid: bool = False,
    ) -> None:
        """Answer single query."""
        run_answer(query, k, index_directory, hybrid)

    def answer_dataset(
        self,
        student_search_results_path: str = "",
        save_directory: str = "data/output/search_results_and_answer",
        index_directory: str = "data/processed",
    ) -> None:
        """Answer dataset and write answers JSON."""
        run_answer_dataset(
            student_search_results_path,
            save_directory,
            index_directory
        )

    def evaluate(
        self, student_search_results_path: str = "", dataset_path: str = ""
    ) -> None:
        """Report Recall@k."""
        run_evaluate(student_search_results_path, dataset_path)
