"""Python splitter: one chunk per top def/class."""  # Докстрінг модуля: один чанк на кожну функцію або клас найвищого рівня.

import ast  # Стандартна бібліотека Python для парсингу синтаксичного дерева коду.
from typing import List  # Анотація типів.

from src.chunk_types import Chunk  # Модель чанка.
from src.chunk_window import spans, split_oversized  # Функції нарізки за межами та ковзного вікна.


def line_offsets(text: str) -> List[int]:
    """Start index of each line."""  # Створює мапу оффсетів: символьний індекс початку кожного рядка.
    out = [0]  # Рядок №1 починається з символу 0.
    # Проходимо по рядках зі збереженням символів переносу:
    for line in text.splitlines(keepends=True):
        # Початок наступного рядка = початок поточного + довжина поточного.
        out.append(out[-1] + len(line))
    return out  # out[i] дає точний індекс початку рядка (i + 1).


def chunk_python(path: str, text: str, max_chunk_size: int = 2000) -> List[Chunk]:
    """Split by top defs/classes, fallback to window."""  # Нарізка Python по верхньорівневих def/class або fallback на вікно.
    try:
        # Намагаємося спарсити код у синтаксичне дерево.
        tree = ast.parse(text)
    except SyntaxError:
        # ЗАХИСТ ВІД СИНТАКСИЧНИХ ПОМИЛОК (Robustness):
        # Якщо файл містить зламаний Python-код (наприклад шаблон чи незавершений скрипт),
        # ми не крашимо програму, а просто ріжемо його як звичайний текст ковзним вікном.
        return split_oversized(path, text, 0, len(text), max_chunk_size)

    # Будуємо масив позицій рядків для швидкого перерахунку "рядок:колонка" -> "абсолютний символ".
    offsets = line_offsets(text)

    # Допоміжна функція: перетворює номер рядка (1-based) і колонку у символьний індекс (0-based).
    def pos(lineno: int, col: int) -> int:
        return offsets[lineno - 1] + col

    bounds = [0]  # Початкова точка файлу.
    
    # Ітеруємося тільки по верхньорівневих вузлах модуля (tree.body):
    for node in tree.body:
        # Нас цікавлять тільки визначення функцій, асинхронних функцій та класів:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue  # Прості оператори (імпорти, змінні) залишаємо в проміжках між функціями.
            
        # Враховуємо декоратори (@dataclass, @staticmethod тощо):
        # Початок вузла — це або перший рядок декоратора, або сам рядок def/class.
        lines = [node.lineno] + [d.lineno for d in node.decorator_list]
        start = pos(min(lines), 0)  # Початок блоку (з нульової колонки рядка).
        
        # Кінець вузла: node.end_lineno та node.end_col_offset вказують на точний кінець тіла функції/класу.
        end = pos(node.end_lineno or node.lineno, node.end_col_offset or 0)
        
        # Якщо між попередньою межею і початком цієї функції є код (наприклад, коментарі чи імпорти):
        if start > bounds[-1]:
            bounds.append(start)  # Додаємо початок функції як окрему точку розділу.
        bounds.append(end)        # Додаємо кінець функції.
        
    bounds.append(len(text))  # Додаємо кінець файлу.
    
    # Видаляємо дублікати, сортуємо точки та перетворюємо інтервали на готові валідні чанки.
    return spans(path, text, sorted(set(bounds)), max_chunk_size)