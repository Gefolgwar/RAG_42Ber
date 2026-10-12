# ==============================================================================
# 42 School: RAG against the machine
# Makefile automating installation, execution, debugging, linting, and cleanup.
# ==============================================================================

PYTHON        := uv run python
UV            := uv
SRC_DIR       := src
API_DIR       := api
RAW_DATA_DIR  := data/raw/vllm-0.10.1
PROCESSED_DIR := data/processed

.PHONY: all install run debug clean lint lint-strict test api help

all: install

# ------------------------------------------------------------------------------
# Dependency Management (Chapter V.2 & V.4)
# Moulinette and peer-evaluators only run `uv sync`
# ------------------------------------------------------------------------------
install:
	@echo "==> Syncing dependencies via uv..."
	$(UV) sync

# ------------------------------------------------------------------------------
# Execution (Chapter V.2 & VI.7.2)
# Executes the default indexing command
# ------------------------------------------------------------------------------
run:
	@echo "==> Running default corpus indexing..."
	$(PYTHON) -m $(SRC_DIR) index --max_chunk_size 2000

# ------------------------------------------------------------------------------
# Debugging (Chapter V.2)
# Runs the main module inside Python's built-in interactive debugger (pdb)
# ------------------------------------------------------------------------------
debug:
	@echo "==> Running in interactive debug mode (pdb)..."
	$(PYTHON) -m pdb -m $(SRC_DIR) index --max_chunk_size 2000

# ------------------------------------------------------------------------------
# Linting & Type Checking (Chapter V.2 & V.4)
# Must pass without errors under flake8 and mypy
# ------------------------------------------------------------------------------
lint:
	@echo "==> Running flake8..."
	$(UV) run flake8 src
	@echo "==> Running mypy with subject-mandated flags..."
	$(UV) run mypy src \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	@echo "==> Running flake8..."
	$(UV) run flake8 src
	@echo "==> Running mypy in strict mode..."
	$(UV) run mypy src --strict

# ------------------------------------------------------------------------------
# Testing
# ------------------------------------------------------------------------------
test:
	@echo "==> Running unit tests..."
	$(UV) run pytest -v

# ------------------------------------------------------------------------------
# Bonus 5: Local REST API Server
# ------------------------------------------------------------------------------
api:
	@echo "==> Starting FastAPI local server on http://localhost:8000..."
	$(UV) run uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# ------------------------------------------------------------------------------
# Cleanup (Chapter V.2)
# Removes bytecode, type checking caches, and processed artifacts
# ------------------------------------------------------------------------------
clean:
	@echo "==> Cleaning temporary caches and bytecode..."
	rm -rf __pycache__ .mypy_cache .pytest_cache .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "==> Cleaning generated output and processed indices..."
	rm -rf $(PROCESSED_DIR)/*
	rm -rf data/output/search_results/*
	rm -rf data/output/search_results_and_answer/*
	@echo "==> Clean complete."

# ------------------------------------------------------------------------------
# Help target
# ------------------------------------------------------------------------------
help:
	@echo "Available Makefile rules:"
	@echo "  make install     : Sync dependencies using uv sync"
	@echo "  make run         : Run default indexing (max_chunk_size=2000)"
	@echo "  make debug       : Launch indexing under pdb debugger"
	@echo "  make lint        : Run flake8 and mypy with subject-mandated flags"
	@echo "  make lint-strict : Run flake8 and mypy in strict mode"
	@echo "  make test        : Run pytest test suite"
	@echo "  make api         : Start local FastAPI server (Bonus 5)"
	@echo "  make clean       : Remove all caches and processed data"