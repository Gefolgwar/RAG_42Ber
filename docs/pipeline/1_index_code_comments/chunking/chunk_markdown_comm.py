"""Markdown splitter: one chunk per header section."""  # Докстрінг модуля: один чанк на кожен розділ із заголовком.

from typing import List  # Анотація типів.

from src.chunk_types import Chunk  # Модель чанка.
from src.chunk_window import spans  # Функція для збирання чанків за знайденими точками розрізу.


def is_header(line: str) -> bool:
    """True for ATX headers (# .. ######)."""  # Перевіряє, чи є рядок валідним ATX-заголовком Markdown.
    stripped = line.lstrip()  # Прибираємо початкові пробіли/табуляції.
    if not stripped.startswith("#"):
        return False  # Якщо рядок не починається з '#', це точно не заголовок.
        
    # Рахуємо, скільки символів '#' стоїть на початку.
    hashes = len(stripped) - len(stripped.lstrip("#"))
    
    # За стандартом Markdown рівні заголовків існують тільки від 1 до 6 (# ... ######).
    if hashes < 1 or hashes > 6:
        return False
        
    # Якщо рядок складається тільки з символів '#' (наприклад, "###\n"), це валідний заголовок.
    if len(stripped) == hashes:
        return True
        
    # За специфікацією Markdown після '#' обов'язково має йти пробіл, таб або перенос рядка:
    # "# Title" — заголовок, але "#tag" — це тег/хештег, а не заголовок!
    return stripped[hashes] in " \t\r\n"


def chunk_markdown(path: str, text: str, max_chunk_size: int = 2000) -> List[Chunk]:
    """Split by headers."""  # Розбиває файл Markdown по заголовках.
    bounds = [0]  # Початкова точка розрізу — самий початок файлу (індекс 0).
    pos = 0       # Лічильник поточної позиції у тексті в символах.
    
    # splitlines(keepends=True) розбиває на рядки, ЗБЕРІГАЮЧИ '\n',
    # щоб сума довжин рядків точно дорівнювала len(text) без зсуву координат!
    for line in text.splitlines(keepends=True):
        # Якщо ми не на початку файлу (pos > 0) і поточний рядок — заголовок:
        if pos > 0 and is_header(line):
            bounds.append(pos)  # Додаємо позицію початку цього рядка як точку нового розділу.
        pos += len(line)        # Зсуваємо позицію на точну довжину рядка з переносом.
        
    bounds.append(len(text))  # Кінцева точка розрізу — кінець усього документу.
    
    # sorted(set(bounds)) прибирає дублікати індексів (якщо такі були) та сортує точки.
    # spans(...) розрізає текст між цими точками та застосовує вікно до завеликих розділів.
    return spans(path, text, sorted(set(bounds)), max_chunk_size)