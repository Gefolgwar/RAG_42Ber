"""Chunk span with exact file offsets."""

from dataclasses import dataclass


@dataclass
class Chunk:
    """Single searchable span."""

    file_path: str
    first_character_index: int
    last_character_index: int
    text: str

    def char_len(self) -> int:
        """Chunk length in characters."""
        return self.last_character_index - self.first_character_index
