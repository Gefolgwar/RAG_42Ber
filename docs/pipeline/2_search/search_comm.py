"""Search commands.

Цей модуль реалізує команди пошуку для CLI (через бібліотеку Python Fire):
1. run_search — пошук джерел для одного запиту (вивід у stdout).
2. run_search_dataset — пакетний пошук по цілому датасету питань із збереженням у JSON.
"""

# Стандартні модулі для роботи з форматом JSON та файловими шляхами
import json
from pathlib import Path

# Бібліотека для відображення прогрес-барів у терміналі (обов'язкова за сабжектом)
from tqdm import tqdm

# Наш дисковий кеш із перевіркою відбитка індексу
from src.cache import QueryCache

# Pydantic-моделі для валідації та структуризації результатів
from src.models import MinimalSearchResults, StudentSearchResults

# Допоміжні функції пошукового рушія з retrieval.py:
# - err: вивід помилок у stderr без падіння програми
# - fingerprint: отримання дати створення індексу з meta.json
# - load_kb: завантаження KnowledgeBase (індексу BM25 та чанків)
# - search_cached: фасад пошуку, який спершу перевіряє кеш, а потім шукає
from src.retrieval import err, fingerprint, load_kb, search_cached


# ============================================================================
# КОМАНДА 1: Пошук для одного запиту
# Викликається через CLI: uv run python -m src search "запит" --k 5
# ============================================================================
def run_search(
    query: str, k: int, index_directory: str, file_type: str, hybrid: bool = False
) -> None:
    """Print top-k sources as JSON list.

    Виконує пошук топ-k найрелевантніших чанків і друкує їх у stdout як JSON.
    """
    try:
        # ПЕРЕВІРКА ГРАНИЧНИХ ВИПАДКІВ (вимога сабжекту — не падати на edge cases):
        # Якщо запит порожній, складається лише з пробілів, або k <= 0:
        if not query or not query.strip() or k <= 0:
            print("[]")  # Повертаємо валідний порожній JSON-масив
            return

        # Валідація фільтра за типом файлів:
        # дозволені лише 'all' (усе), 'docs' (документація) та 'code' (Python-код)
        if file_type not in ("all", "docs", "code"):
            err(f"invalid file_type={file_type!r}")
            print("[]")
            return

        # Завантажуємо пошуковий індекс KnowledgeBase із директорії (за замовчуванням data/processed)
        kb = load_kb(index_directory)
        if kb is None:
            # Якщо індекс не знайдено або він пошкоджений (помилка вже виведена у stderr через err)
            print("[]")
            return

        # Ініціалізуємо дисковий кеш:
        # fingerprint(...) читає дату 'created_at' з data/processed/meta.json.
        # Якщо індекс переіндексували — старий кеш автоматично очиститься.
        cache = QueryCache(fingerprint=fingerprint(index_directory))

        # Виконуємо пошук через шар кешу:
        # Якщо цей запит уже робили — миттєво поверне джерела з cache.get().
        # Якщо ні — запустить BM25 (або Hybrid), поверне результат і збереже у cache.put().
        sources = search_cached(kb, query, k, file_type, cache, index_directory, hybrid)

        # Конвертуємо знайдені Pydantic-об'єкти MinimalSource у звичайні словники (.model_dump())
        # та виводимо у консоль як JSON-рядок (stdout)
        print(json.dumps([s.model_dump() for s in sources]))

    except Exception as exc:
        # Захист від аварійного завершення (unhandled traceback):
        # Сабжект вимагає, щоб CLI ніколи не крашився під час оцінювання мулінеткою.
        err(f"search failed: {exc}")
        print("[]")


# ============================================================================
# КОМАНДА 2: Пакетний пошук по датасету питань
# Викликається через CLI: uv run python -m src search_dataset --dataset_path ... --k 10
# ============================================================================
def run_search_dataset(
    dataset_path: str,
    k: int,
    save_directory: str,
    index_directory: str,
    file_type: str,
    hybrid: bool = False,
) -> None:
    """Search dataset and write results JSON.

    Зчитує JSON-файл із питаннями, виконує пошук для кожного питання,
    формує структуру StudentSearchResults і зберігає її на диск.
    """
    try:
        # 1. Завантажуємо індекс у пам'ять один раз перед обробкою всього датасету
        kb = load_kb(index_directory)
        if kb is None:
            return  # Якщо індексу немає — перериваємо роботу

        # 2. Безпечно читаємо вхідний файл датасету
        try:
            data = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            # Обробка випадків, якщо файл не існує або містить битий JSON
            err(f"cannot read dataset '{dataset_path}': {exc}")
            return

        # Витягуємо список запитань із поля 'rag_questions' (формат за стандартом RagDataset)
        questions = data.get("rag_questions", []) if isinstance(data, dict) else []

        # 3. Створюємо екземпляр кешу з відбитком поточного індексу
        cache = QueryCache(fingerprint=fingerprint(index_directory))

        # Список для накопичення результатів по кожному питанню
        results = []

        # 4. Проходимо по всіх питаннях за допомогою tqdm (індикатор прогресу в терміналі)
        for q in tqdm(questions, desc="Searching", unit="q"):
            try:
                # Отримуємо ID питання та сам текст запитання
                qid = str(q.get("question_id", ""))
                text = str(q.get("question", ""))

                # Якщо текст питання є — шукаємо джерела (з використанням кешу)
                if text:
                    srcs = search_cached(
                        kb, text, k, file_type, cache, index_directory, hybrid
                    )
                else:
                    # Якщо питання порожнє — повертаємо порожній список джерел
                    srcs = []

                # Формуємо об'єкт MinimalSearchResults для поточного питання
                # і конвертуємо його у словник через .model_dump()
                results.append(MinimalSearchResults(
                    question_id=qid, question=text,
                    retrieved_sources=srcs).model_dump())

            except Exception as exc:
                # Якщо одне конкретне питання має пошкоджену структуру — пропускаємо його,
                # щоб не переривати обробку всього іншого датасету
                err(f"skip question: {exc}")

        # 5. Упаковуємо всі зібрані результати в фінальну Pydantic-модель StudentSearchResults,
        # яка суворо відповідає специфікації Chapter VI.4 та VI.5 сабжекту
        out = StudentSearchResults(
            search_results=[MinimalSearchResults(**r) for r in results], k=k)

        # 6. Створюємо вихідну папку, якщо її ще немає (наприклад, data/output/search_results/...)
        save_root = Path(save_directory)
        save_root.mkdir(parents=True, exist_ok=True)

        # Формуємо кінцевий шлях до файлу (зберігаємо ту саму назву файлу, що й у датасету)
        dest = save_root / Path(dataset_path).name

        # Записуємо згенерований результат у файл у форматі красивого JSON (indent=2)
        dest.write_text(out.model_dump_json(indent=2), encoding="utf-8")

        # Інформуємо користувача про успішне збереження файлу
        print(f"Saved student_search_results to {dest.as_posix()}")

    except Exception as exc:
        # Загальний перехват помилок команди search_dataset для уникнення падіння процесу
        err(f"search_dataset failed: {exc}")