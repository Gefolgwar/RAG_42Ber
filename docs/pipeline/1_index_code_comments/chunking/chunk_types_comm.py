"""Chunk span with exact file offsets."""  # Докстрінг модуля: описує діапазон чанка з точними символьними оффсетами.

from dataclasses import dataclass  # Використовуємо стандартний dataclass замість Pydantic для внутрішніх структур, бо він значно швидший і легший у пам'яті.


@dataclass
class Chunk:
    """Single searchable span."""  # Описує один неподільний фрагмент тексту для індексу.

    file_path: str               # Шлях до файлу відносно кореня проєкту (наприклад, "data/raw/vllm-0.10.1/docs/...").
    first_character_index: int   # Початковий індекс символу у вихідному файлі (включно, початок інтервалу).
    last_character_index: int    # Кінцевий індекс символу у вихідному файлі (не включно, кінець інтервалу).
    text: str                    # Сам текстовий фрагмент, що відповідає зрізу file_content[first:last].

    def char_len(self) -> int:
        """Chunk length in characters."""  # Допоміжний метод: повертає точну довжину чанка в символах.
        return self.last_character_index - self.first_character_index  # Різниця координат гарантує точний розмір зрізу.