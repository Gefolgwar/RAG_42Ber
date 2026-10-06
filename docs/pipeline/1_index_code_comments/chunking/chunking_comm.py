"""Chunking facade: single seam for callers."""  # Докстрінг модуля: визначає файл як єдиний фасад (шлюз) для розбиття файлів на чанки.

from typing import List  # Імпорт типу List для статичної типізації mypy (вимога 42).

# Імпортуємо два спеціалізовані сплітери та базовий клас Chunk:
from src.chunk_markdown import chunk_markdown  # Алгоритм нарізки документації за заголовками (#, ##...).
from src.chunk_python import chunk_python      # Алгоритм нарізки коду Python за синтаксичним деревом AST.
from src.chunk_types import Chunk              # Датаклас, що описує один чанк із координатами.

# __all__ визначає публічний інтерфейс модуля: що саме імпортуватиметься при виклику `from src.chunking import *`.
__all__ = ["Chunk", "chunk_python", "chunk_markdown", "chunk_file"]


def chunk_file(path: str, text: str, max_chunk_size: int = 2000) -> List[Chunk]:
    """Split by extension: .py -> AST, else headers."""  # Докстрінг функції з описом логіки вибору сплітера.

    # Перевіряємо розширення файлу: якщо це скрипт Python:
    if path.endswith(".py"):
        # Передаємо в синтаксичний сплітер Python AST, який виділяє функції та класи цілісними блоками.
        return chunk_python(path, text, max_chunk_size)

    # Якщо це не .py (тобто .md, .markdown, .rst, .txt) — розбиваємо за структурою заголовків Markdown.
    return chunk_markdown(path, text, max_chunk_size)


# Блок для прямого тестування модуля: запускається тільки тоді, коли файл виконується безпосередньо (python chunking.py),
# а не через import з іншого місця.
if __name__ == "__main__":
    print("=== [STEP 5 DIAGNOSTIC] Testing Python AST & Markdown Splitters ===")

    # 1. Створюємо тестовий шматочок коду Python з функцією foo().
    py_sample = "import os\n\n\ndef foo():\n    x = 1\n    return x\n\n\nprint(foo())\n"
    
    # Нарізаємо його з жорстким лімітом у 100 символів на чанк.
    py_chunks = chunk_python("a.py", py_sample, 100)
    print(f"[Python AST Chunking]: Generated {len(py_chunks)} chunks:")
    
    # Перевіряємо згенеровані чанки Python:
    for i, ch in enumerate(py_chunks, 1):
        # САМА ВАЖЛИВА ПЕРЕВІРКА: зріз оригінального рядка за збереженими оффсетами
        # ПОВИНЕН посимвольно збігатися з текстом чанка!
        assert py_sample[ch.first_character_index:ch.last_character_index] == ch.text
        # Довжина не має перевищувати заданий ліміт (100 символів).
        assert ch.char_len() <= 100
        
        # Готуємо красиве прев'ю тексту чанка (замінюємо переноси рядків на \n).
        preview = ch.text.replace("\n", "\\n")[:40]
        print(f"  Chunk #{i} [{ch.first_character_index}:{ch.last_character_index}]"
              f" ({ch.char_len()} chars): {preview}")
        print("  [Verification]: Direct string slice matches chunk text: OK!")

    # 2. Створюємо тестовий документ Markdown із заголовками # та ##.
    md_sample = "Intro line\n\n# Title\ntext under title\n\n## Next\nmore text\n"
    
    # Нарізаємо його за заголовками з тим самим лімітом 100 символів.
    md_chunks = chunk_markdown("b.md", md_sample, 100)
    print(f"[Markdown Header Chunking]: Generated {len(md_chunks)} chunks:")
    
    # Перевіряємо чанки Markdown аналогічним чином:
    for ch in md_chunks:
        # Перевірка збігу зрізу тексту за індексами:
        assert md_sample[ch.first_character_index:ch.last_character_index] == ch.text
        # Перевірка дотримання ліміту розміру:
        assert ch.char_len() <= 100
        print("  [Verification]: Direct string slice matches chunk text: OK!")