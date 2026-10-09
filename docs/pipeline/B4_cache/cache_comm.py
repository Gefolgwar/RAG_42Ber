"""Disk cache for repeated queries (Bonus 4).

Модуль реалізує дискове кешування результатів пошуку з механізмом
автоматичної інвалідації при зміні пошукового індексу.
"""

import hashlib
import json
from pathlib import Path
from typing import List, Optional

# Модель Pydantic для представлення відрізка знайденого тексту
from src.models import MinimalSource

# Стандартний шлях до файлу кешу на диску:
# Використовується за замовчуванням у конструкторі QueryCache
CACHE_FILE = "data/processed/query_cache.json"


# ============================================================================
# ФУНКЦІЯ: cache_key
# ============================================================================
# ДЕ ВИКЛИКАЄТЬСЯ В ПРОЄКТІ:
# - Тільки локально всередині цього ж файлу (src/cache.py):
#     1) У методі QueryCache.get(...) для пошуку ключа.
#     2) У методі QueryCache.put(...) для збереження запису.
# ============================================================================
def cache_key(query: str, k: int, hybrid: bool = False) -> str:
    """Build SHA-256 key for normalized query plus k.

    Генерує детермінований унікальний SHA-256 хеш для комбінації
    запиту, кількості результатів (k) та типу пошуку (hybrid/lexical).
    """
    # 1. Нормалізація: обрізаємо пробіли з країв та переводимо текст у нижній регістр,
    #    щоб "LoRA", "lora " та " lora" давали один і той самий результат у кеші.
    #    Формуємо рядок вигляду: "how to configure:5"
    norm = f"{query.strip().lower()}:{k}"

    # 2. Якщо ввімкнено гібридний пошук (BM25 + Dense векторний),
    #    додаємо суфікс ":hybrid", щоб результати гібридного пошуку
    #    не перетирали результати звичайного BM25 пошуку.
    #    Вийде, наприклад: "how to configure:5:hybrid"
    if hybrid:
        norm += ":hybrid"

    # 3. Перетворюємо рядок у байти (UTF-8) та рахуємо хеш SHA-256 (64 hex-символи).
    #    errors="replace" захищає від збоїв при наявності невалідних символів.
    return hashlib.sha256(norm.encode("utf-8", errors="replace")).hexdigest()


# ============================================================================
# КЛАС: QueryCache
# ============================================================================
# ДЕ ІМПОРТУЄТЬСЯ В ПРОЄКТІ:
# - src/commands/search_cmd.py: для створення екземпляра кешу перед пошуком.
# - src/retrieval.py: для анотації типу аргументу `cache: QueryCache` у `search_cached`.
# ============================================================================
class QueryCache:
    """Persistent dict cache with index-fingerprint invalidation.

    Клас керує кешем у пам'яті (_data) та синхронізацією з файлом на диску.
    Автоматично очищує кеш, якщо індекс було перезібрано.
    """

    # ------------------------------------------------------------------------
    # МЕТОД: __init__
    # ------------------------------------------------------------------------
    # ДЕ ВИКЛИКАЄТЬСЯ В ПРОЄКТІ:
    # 1) src/commands/search_cmd.py -> run_search():
    #      cache = QueryCache(fingerprint=fingerprint(index_directory))
    # 2) src/commands/search_cmd.py -> run_search_dataset():
    #      cache = QueryCache(fingerprint=fingerprint(index_directory))
    # ------------------------------------------------------------------------
    def __init__(self, path: str = CACHE_FILE, fingerprint: str = "") -> None:
        """Load cache, clearing it when the index fingerprint changed.

        Args:
            path: Шлях до JSON-файлу кешу на диску.
            fingerprint: Унікальний відбиток поточного індексу (timestamp
                         'created_at' із файлу meta.json).
        """
        self.path = path
        self.fingerprint = fingerprint

        # За замовчуванням ініціалізуємо чисту структуру кешу в пам'яті
        self._data: dict[str, object] = {"fingerprint": fingerprint, "items": {}}

        try:
            # Спроба зчитати вже наявний файл кешу з диска
            raw = json.loads(Path(path).read_text(encoding="utf-8"))

            # ВАЖЛИВИЙ МОМЕНТ ІНВАЛІДАЦІЇ:
            # Кеш завантажується лише тоді, коли fingerprint з файлу точно збігається
            # з поточним fingerprint індексу. Якщо користувач запустив переіндексацію,
            # 'created_at' у meta.json зміниться, умова не виконається, і старий кеш
            # буде скинуто (перезаписано новим чистим станом).
            if isinstance(raw, dict) and raw.get("fingerprint") == fingerprint:
                self._data = raw
        except OSError:
            # Файлу ще не існує (перший запуск) або немає прав читання — ігноруємо
            pass
        except ValueError:
            # Файл пошкоджений або містить невалідний JSON — ігноруємо і почнемо з нуля
            pass

    # ------------------------------------------------------------------------
    # МЕТОД: get
    # ------------------------------------------------------------------------
    # ДЕ ВИКЛИКАЄТЬСЯ В ПРОЄКТІ:
    # 1) src/retrieval.py -> search_cached():
    #      hit = cache.get(query, k, hybrid)
    #      if hit is not None:
    #          return hit[: max(k, 0)]
    # ------------------------------------------------------------------------
    def get(self, query: str, k: int, hybrid: bool = False) -> Optional[List[MinimalSource]]:
        """Return cached sources or None on miss.

        Повертає список об'єктів MinimalSource при збігу (Cache Hit)
        або None, якщо запиту немає в кеші (Cache Miss).
        """
        try:
            # Отримуємо словник збережених запитів
            items = self._data.get("items")
            assert isinstance(items, dict)

            # Шукаємо запис за хешем запиту
            hit = items.get(cache_key(query, k, hybrid))
            if not isinstance(hit, list):
                # Запиту немає в кеші (Miss)
                return None

            # Якщо знайшли — десеріалізуємо сирі dict назад у моделі Pydantic MinimalSource
            return [MinimalSource(**s) for s in hit if isinstance(s, dict)]
        except (ValueError, TypeError, AssertionError):
            # Якщо дані в кеші пошкоджені чи мають невірний формат — вважаємо це промахом
            return None

    # ------------------------------------------------------------------------
    # МЕТОД: put
    # ------------------------------------------------------------------------
    # ДЕ ВИКЛИКАЄТЬСЯ В ПРОЄКТІ:
    # 1) src/retrieval.py -> search_cached():
    #      # Викликається після того, як BM25 або Hybrid пошук повернув результати:
    #      cache.put(query, k, sources, hybrid)
    # ------------------------------------------------------------------------
    def put(
        self, query: str, k: int, sources: List[MinimalSource], hybrid: bool = False
    ) -> None:
        """Store sources and persist cache to disk.

        Зберігає знайдені джерела у пам'ять і синхронізує весь кеш у JSON-файл.
        """
        try:
            # 1. Переконуємося, що поле "items" існує і є словником
            items = self._data.get("items")
            if not isinstance(items, dict):
                items = {}
                self._data["items"] = items

            # 2. Серіалізуємо об'єкти MinimalSource у прості словники через model_dump()
            #    та записуємо в items під SHA-256 хеш-ключем
            items[cache_key(query, k, hybrid)] = [s.model_dump() for s in sources]

            # 3. Створюємо папку (data/processed/), якщо вона ще не створена
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)

            # 4. Атомарно перезаписуємо весь файл query_cache.json на диску
            Path(self.path).write_text(json.dumps(self._data), encoding="utf-8")
        except OSError:
            # Захист: якщо сталася помилка запису на диск (диск переповнений,
            # немає прав), програма не падає, а продовжує роботу
            pass