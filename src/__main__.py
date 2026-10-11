"""Entry point for `python -m src <command>` (Chapter VI.6)."""

import fire  # type: ignore[import-untyped]

from src.cli import CLI


def main() -> None:
    """Dispatch CLI commands via Python Fire."""
    fire.Fire(CLI)


if __name__ == "__main__":
    main()
