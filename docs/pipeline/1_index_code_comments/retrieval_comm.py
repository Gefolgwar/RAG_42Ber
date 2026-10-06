"""Single seam for search: cache, BM25, hybrid, spans."""  # Докстрінг модуля: описує єдину точку входу для пошукової логіки.

import json  # Модуль для парсингу файлів конфігурації/метаданих (наприклад, meta.json).
import sys  # Потрібен для доступу до потоку sys.stderr при виведенні діагностичних повідомлень.
from pathlib import Path  # Зручна і кросплатформна робота зі шляхами файлової системи.
from typing import Dict, List, Optional, Tuple  # Анотації типів для статичного аналізу через mypy.

from src.cache import QueryCache  # Клас дискового кешу для збереження результатів частих запитів (Bonus 4).
from src.chunking import Chunk  # Внутрішня модель чанка з текстом та символьними оффсетами.
from src.index import KnowledgeBase  # Клас збереженого BM25 індексу корпусу.
from src.ingest import read_file_safely  # Функція безпечного читання файлу без аварійного завершення на бінарних даних.
from src.models import MinimalSource  # Pydantic-модель зрізу (джерела), яку вимагає сабжект для вихідного JSON.


def err(msg: str) -> None:
    """Print error to stderr."""  # Виводить повідомлення про помилку в потік stderr.
    print(f"Error: {msg}", file=sys.stderr)  # Завдяки цьому stdout лишається чистим (JSON виводиться тільки в stdout).


def load_kb(index_directory: str) -> Optional[KnowledgeBase]:
    """Load index or report error."""  # Завантажує KnowledgeBase з диску або повертає None при збої.
    try:
        return KnowledgeBase.load(index_directory)  # Відновлює індекс BM25 та список чанків з data/processed.
    except Exception as exc:
        err(f"cannot load index from '{index_directory}': {exc}")  # Логуємо помилку в stderr.
        return None  # Повертаємо None замість аварійного падіння, щоб CLI працював стабільно.


def fingerprint(index_directory: str) -> str:
    """Read index fingerprint for cache invalidation."""  # Читає мітку часу створення індексу для інвалідації кешу.
    try:
        # Зчитуємо meta.json, збережений під час останнього індексування.
        meta = json.loads((Path(index_directory) / "meta.json").read_text("utf-8"))
        # Повертаємо ISO-рядок created_at: якщо індекс перебудують, рядок зміниться, і старий кеш стане недійсним.
        return str(meta.get("created_at", ""))
    except (OSError, ValueError):
        return ""  # Якщо файл відсутній або пошкоджений, повертаємо порожній відбиток.


def chunk_to_source(ch: Chunk) -> MinimalSource:
    """Convert Chunk to output model."""  # Конвертує внутрішній об'єкт Chunk у Pydantic-модель MinimalSource.
    return MinimalSource(
        file_path=ch.file_path,  # Шлях до файлу відносно кореня проєкту.
        first_character_index=ch.first_character_index,  # Початок зрізу символів.
        last_character_index=ch.last_character_index,  # Кінець зрізу символів.
    )  # Текст тут відкидається, бо за схемою сабжекту MinimalSource містить лише шлях і координати.


def chunk_key(ch: Chunk) -> str:
    """Dedup key for a chunk."""  # Формує унікальний ідентифікатор для дедуплікації чанка.
    return f"{ch.file_path}:{ch.first_character_index}"  # Файл + початкова координата однозначно ідентифікують фрагмент.


def hybrid_search(
    kb: KnowledgeBase, query: str, k: int, file_type: str, index_dir: str
) -> List[Tuple[Chunk, float]]:
    """Fuse BM25 and dense with RRF, fallback to BM25."""  # Поєднує лексичний (BM25) та семантичний пошук через RRF (Bonus 2).
    # 1. Отримуємо ранжовані результати класичного лексичного пошуку BM25.
    bm25_pairs = kb.search_bm25(query, k=k, file_type=file_type)
    try:
        # Ленивий імпорт модулів щільного векторного пошуку (щоб не тягнути PyTorch/Transformers без потреби).
        from src.dense import DenseIndex, rrf_fuse

        # Завантажуємо векторний індекс ембедінгів із зазначеної директорії.
        dense = DenseIndex.load(str(Path(index_dir) / "dense_index"))
        dense.chunks = kb.chunks  # Синхронізуємо об'єкти чанків із лексичною базою.
        
        # Отримуємо результати косинусного пошуку ембедінгів.
        dense_pairs = dense.search_dense(query, k=k, file_type=file_type)
        if not dense_pairs:
            return bm25_pairs  # Якщо векторний пошук порожній, віддаємо просто BM25.

        # Словник для швидкого доступу до пари (Chunk, score) за згенерованим ключем чанка.
        by_key: Dict[str, Tuple[Chunk, float]] = {}
        for ch, sc in bm25_pairs + dense_pairs:
            by_key.setdefault(chunk_key(ch), (ch, sc))  # Зберігаємо перший знайдений екземпляр.

        # Об'єднуємо два ранжовані списки алгоритмом Reciprocal Rank Fusion (RRF).
        fused = rrf_fuse(
            [[chunk_key(ch) for ch, _ in bm25_pairs],
             [chunk_key(ch) for ch, _ in dense_pairs]],
            k=k,
        )
        # Повертаємо об'єкти Chunk із перерахованими балами RRF.
        return [(by_key[key][0], score) for key, score in fused if key in by_key]
    except Exception:
        # Якщо семантичний індекс не знайдено чи сталася помилка — відкочуємося до надійного BM25.
        return bm25_pairs


def search_cached(
    kb: KnowledgeBase,
    query: str,
    k: int,
    file_type: str,
    cache: QueryCache,
    index_dir: str,
    hybrid: bool = False,
) -> List[MinimalSource]:
    """Search with cache, store on miss."""  # Виконує пошук із підтримкою дискового кешування (Bonus 4).
    # 1. Перевіряємо, чи є вже збережена відповідь для цієї пари (query, k).
    hit = cache.get(query, k)
    if hit is not None:
        return hit[: max(k, 0)]  # Cache HIT: миттєво віддаємо результат без повторних обчислень!

    # 2. Cache MISS: виконуємо реальний пошук.
    if hybrid:
        pairs = hybrid_search(kb, query, k, file_type, index_dir)
    else:
        pairs = kb.search_bm25(query, k=k, file_type=file_type)

    # 3. Перетворюємо знайдені чанки на об'єкти MinimalSource.
    sources = [chunk_to_source(ch) for ch, _ in pairs]
    
    # 4. Записуємо результат у кеш для наступних викликів.
    cache.put(query, k, sources)
    return sources


# Глобальний кеш у пам'яті: зберігає мапу чанків за їхніми координатами, щоб не парсити диск повторно.
_SPAN_CACHE: Dict[str, Dict[Tuple[str, int], Chunk]] = {}


def _span_map(index_directory: str) -> Dict[Tuple[str, int], Chunk]:
    """Load chunk span map once per index directory (CPU fast path)."""  # Будує швидку хеш-таблицю чанків у RAM.
    hit = _SPAN_CACHE.get(index_directory)
    if hit is not None:
        return hit  # Якщо таблиця вже в пам'яті — повертаємо одразу (0 overhead на диск).
    try:
        # Завантажуємо чанки з індексу, створеного під час кроку index.
        kb = KnowledgeBase.load(index_directory)
        # Будуємо словник: ключ — кортеж (file_path, first_character_index), значення — сам об'єкт Chunk.
        by_span: Dict[Tuple[str, int], Chunk] = {
            (c.file_path, c.first_character_index): c for c in kb.chunks
        }
    except Exception:
        by_span = {}  # Запобіжник: якщо завантаження не вдалося, повертаємо порожній словник.
    _SPAN_CACHE[index_directory] = by_span  # Кешуємо словник у модулі.
    return by_span


def resolve_chunks(sources: List[MinimalSource], index_directory: str) -> List[Chunk]:
    """Map sources back to texts via index or disk."""  # Відновлює текст чанка за його координатами (потрібно для генерації відповіді).
    # Отримуємо таблицю швидкого доступу з пам'яті.
    by_span = _span_map(index_directory)
    out: List[Chunk] = []
    
    for src in sources:
        # 1. Швидкий шлях: намагаємося взяти готовий чанк із пам'яті індексу.
        hit = by_span.get((src.file_path, src.first_character_index))
        if hit is not None and hit.last_character_index == src.last_character_index:
            out.append(hit)  # Знайдено точний збіг — додаємо в результат.
            continue
            
        # 2. Повільний резервний шлях (fallback): якщо чанка немає в індексі або діапазон інший,
        # читаємо файл безпосередньо з диску та робимо зріз тексту за індексами [first:last].
        text = read_file_safely(src.file_path)
        span = text[src.first_character_index:src.last_character_index]
        out.append(Chunk(src.file_path, src.first_character_index,
                         src.last_character_index, span))
                         
    return out  # Повертаємо список повноцінних Chunk із заповненим полем .text.