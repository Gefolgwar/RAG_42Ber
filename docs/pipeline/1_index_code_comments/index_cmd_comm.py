"""Build chunk index."""  # Докстрінг модуля згідно з PEP 257, описує призначення файлу.

import json  # Модуль для збереження/читання метаданих, хешів та чанків у форматі JSON.
from pathlib import Path  # Об'єктно-орієнтований інтерфейс для зручної роботи зі шляхами файлової системи.
from typing import List  # Анотація типів: для статичної перевірки mypy (List[Chunk]).

from tqdm import tqdm  # Бібліотека для відображення прогрес-барів у терміналі (вимога сабжекту для довгих операцій).

from src.chunking import Chunk, chunk_file  # Клас чанка та фасад-функція, яка обирає між Markdown та Python чанкерами.
from src.hash import diff_hashes, hashes_for  # Допоміжні функції для інкрементальної індексації (Bonus 3).
from src.index import KnowledgeBase  # Клас нашої бази знань, що створює та зберігає BM25 індекс.
from src.ingest import collect_files  # Функція для рекурсивного сканування та безпечного зчитування файлів.
from src.retrieval import err  # Функція для виводу помилок у стандартний потік stderr замість аварійного падіння.


def run_index(
    repo_path: str = "data/raw/vllm-0.10.1",  # Шлях до директорії сирих файлів репозиторію.
    save_directory: str = "data/processed",   # Папка, куди зберігатимуться готові індекси та кеші.
    max_chunk_size: int = 2000,              # Максимальний розмір одного чанка в символах (за сабжектом дефолт 2000).
    dense: bool = False,                     # Прапорець для побудови векторного ембедінг-індексу (Bonus 1).
) -> None:
    """Chunk corpus and persist index."""  # Докстрінг функції згідно з вимогами оформлення 42.

    # 1. Валідація: розмір чанка має бути строго цілим числом у межах від 1 до 2000 символів.
    # Сабжект суворо забороняє чанки > 2000 символів (moulinette забракує весь результат).
    if not isinstance(max_chunk_size, int) or not 0 < max_chunk_size <= 2000:
        err(f"invalid max_chunk_size={max_chunk_size!r}: need 0 < v <= 2000")
        return  # Безпечно виходимо без крашу програми (вимога: handle degenerate inputs gracefully).

    # 2. Валідація: перевіряємо, чи існує вихідна папка з репозиторієм на диску.
    if not Path(repo_path).is_dir():
        err(f"repo path not found: '{repo_path}'")
        return  # Якщо папки немає, виводимо помилку в stderr і коректно завершуємо роботу.

    try:
        # Створюємо об'єкт шляху для цільової директорії та створюємо її, якщо вона ще не існує (разом із батьківськими).
        save_root = Path(save_directory)
        save_root.mkdir(parents=True, exist_ok=True)

        # Спроба зчитати хеші файлів від попереднього запуску індексації (для Bonus 3 — Incremental indexing).
        try:
            old_hashes = json.loads((save_root / "hashes.json").read_text("utf-8"))
        except (OSError, ValueError):
            old_hashes = {}  # Якщо файлу немає або він пошкоджений, вважаємо, що попередніх хешів не було.

        # Зчитуємо всі файли репозиторію (повертає список кортежів: (відносний_шлях, текст_файлу)).
        files = collect_files(repo_path, relative_to=".", file_type="all")
        
        # Рахуємо SHA-256 хеші для поточного стану файлів на диску.
        cur_hashes = hashes_for(files)
        
        # Знаходимо різницю: які файли додалися/змінилися (`changed`), а які були видалені (`deleted`).
        changed, deleted = diff_hashes(old_hashes, cur_hashes)

        # Спроба завантажити вже готові чанки з попереднього індексу.
        try:
            raw = json.loads((save_root / "chunks.json").read_text("utf-8"))
            old_chunks = [Chunk(**r) for r in raw if isinstance(r, dict)]
        except (OSError, ValueError, TypeError):
            old_chunks = []  # Якщо файлу chunks.json нема, починаємо з порожнього списку.

        # ОПТИМІЗАЦІЯ ШВИДКОДІЇ:
        # Якщо вже є чанки, папка bm25_index існує, і жоден файл НЕ змінився та НЕ видалився:
        if old_chunks and (save_root / "bm25_index").is_dir():
            if not changed and not deleted:
                print(f"Index up-to-date! {len(old_chunks)} chunks under {save_directory}/")
                return  # Не витрачаємо час — індекс уже актуальний. Завершуємо за мілісекунди!

        # Фільтруємо старі чанки: залишаємо лише ті, чиї файли НЕ змінювалися і НЕ були видалені.
        keep = []
        for c in old_chunks:
            if c.file_path not in changed:
                if c.file_path not in deleted:
                    keep.append(c)

        # Створюємо швидкий словник для миттєвого доступу до тексту файлу за його шляхом: {шлях: текст}.
        by_path = {p: t for p, t in files}
        
        # Починаємо збирати фінальний список чанків, додаючи збережені валідні старі чанки.
        chunks: List[Chunk] = list(keep)
        
        # Визначаємо, які файли треба нарізати:
        # Якщо у нас інкрементальне оновлення — лише `changed`. Якщо це перший запуск — абсолютно всі файли (`by_path`).
        targets = sorted(changed) if old_chunks else sorted(by_path)
        
        # Запускаємо цикл нарізки файлів із відображенням прогрес-бару в терміналі через tqdm.
        for path in tqdm(targets, desc="Chunking", unit="file"):
            text = by_path.get(path)
            if text is None:
                continue  # Пропускаємо, якщо тексту немає.
            try:
                # Нарізаємо файл на чанки згідно з його розширенням (AST для .py, заголовки для .md)
                # та додаємо згенеровані чанки до загального списку.
                chunks.extend(chunk_file(path, text, max_chunk_size))
            except Exception as exc:
                # Якщо окремий файл містить синтаксичну помилку або збій — не крашимо всю програму,
                # а логуємо помилку і йдемо до наступного файлу.
                err(f"skip '{path}': {exc}")

        # Ініціалізуємо клас нашої бази знань усіма зібраними чанками.
        kb = KnowledgeBase(chunks)
        
        # Виводимо повідомлення без ламання прогрес-бару tqdm.
        tqdm.write("Building BM25 index...")
        
        # Токенізуємо тексти чанків і будуємо структуру зворотного індексу BM25.
        kb.build()
        
        # Зберігаємо BM25 індекс, словник токенів та chunks.json у цільову папку.
        kb.save(save_directory)
        
        # Оновлюємо hashes.json на диску, щоб наступний запуск знав новий стан файлів.
        (save_root / "hashes.json").write_text(
            json.dumps(cur_hashes, indent=2), encoding="utf-8")

        # Опційний функціонал (Bonus 1 / 2): побудова векторних ембедінгів MiniLM.
        if dense:
            try:
                from src.dense import DenseIndex

                idx = DenseIndex(chunks)  # Створюємо об'єкт векторного індексу.
                idx.build()               # Проганяємо тексти чанків через нейромережу all-MiniLM-L6-v2.
                idx.save(str(save_root / "dense_index"))  # Зберігаємо .npy матрицю на диск.
            except Exception as exc:
                err(f"dense index skipped: {exc}")  # Якщо щось пішло не так — не крашимо базовий індекс.

        # Фінальне повідомлення про успіх операції із зазначенням кількості проіндексованих чанків.
        print(f"Ingestion complete! Indexed {len(chunks)} chunks under {save_directory}/")

    except Exception as exc:
        # Глобальний запобіжник: жодна непередбачена помилка не повинна крашити CLI з незрозумілим traceback.
        err(f"indexing failed: {exc}")