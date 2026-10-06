"""Sliding window for oversized spans."""  # Докстрінг модуля: реалізує ковзне вікно для фрагментів, що перевищують ліміт.

from typing import List  # Анотація типів для списків (вимога mypy).

from src.chunk_types import Chunk  # Імпорт базового класу чанка.


def split_oversized(path: str, text: str, start: int, end: int, max_size: int) -> List[Chunk]:
    """Split [start:end) with 10% overlap."""  # Нарізає завеликий діапазон [start:end) вікнами з 10% перекриттям.
    out: List[Chunk] = []  # Список результуючих чанків.
    
    # Крок зсуву вікна (step): беремо розмір вікна мінус 10% на перекриття.
    # Наприклад, для 2000 символів крок буде 2000 - 200 = 1800 символів.
    # max(..., 1) захищає від зациклення, якщо max_size передадуть дуже малим.
    step = max(max_size - max_size // 10, 1)
    
    pos = start  # Поточна позиція початку вікна.
    while pos < end:
        # Кінець вікна — або pos + max_size, або сам кінець тексту (end), якщо залишився менший хвостик.
        stop = min(pos + max_size, end)
        
        # Створюємо чанк із точним зрізом тексту за поточними координатами.
        out.append(Chunk(path, pos, stop, text[pos:stop]))
        
        # Якщо ми вже досягли або покрили кінець фрагмента — виходимо з циклу.
        if stop >= end:
            break
            
        # Зсуваємо вікно вперед на крок зі збереженням перекриття.
        pos += step
        
    return out  # Повертаємо список нарізаних під-чанків.


def spans(path: str, text: str, bounds: List[int], max_size: int) -> List[Chunk]:
    """Build chunks from sorted bounds."""  # Формує чанки між точками розрізу (bounds) із перевіркою розміру.
    out: List[Chunk] = []  # Список чанків.
    
    # zip(bounds, bounds[1:]) бере сусідні межі: (bounds[0], bounds[1]), (bounds[1], bounds[2]) тощо.
    for start, end in zip(bounds, bounds[1:]):
        # Якщо логічний блок (розділ чи функція) вміщується в ліміт (<= max_size):
        if end - start <= max_size:
            # Зберігаємо його цілком одним шматком без розривів.
            out.append(Chunk(path, start, end, text[start:end]))
        else:
            # Якщо блок гігантський (> max_size) — нарізаємо його ковзним вікном.
            out.extend(split_oversized(path, text, start, end, max_size))
            
    return out  # Повертаємо готовий список валідних чанків.