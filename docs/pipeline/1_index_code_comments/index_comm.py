"""BM25 knowledge base with persistence (Step 6).

Chain: chunks -> "file_path + text" -> bm25s -> top-k search.
Layout: chunks.json + meta.json + bm25_index/ under save directory.
"""  # Докстрінг модуля: описує конвеєр пошуку та структуру файлів на диску в data/processed/.

import json  # Для збереження чанків (chunks.json) та метаінформації (meta.json).
from datetime import datetime, timezone  # Для генерації мітки часу UTC (відбиток індексу для кешу).
from pathlib import Path  # Для об'єктної роботи з директоріями та файлами.
from typing import Any, List, Optional, Tuple  # Анотація типів для static type checker mypy.

import bm25s  # Високопродуктивна бібліотека лексичного пошуку BM25 на мові Python/C.

from src.chunking import Chunk  # Клас чанка (шлях, початок, кінець, текст).
from src.file_filter import matches  # Функція фільтрації (all / docs / code).
from src.tokenizer import build_tokenizer  # Функція створення токенізатора з попереднього модуля.


class KnowledgeBase:
    """Chunk collection with BM25 index, save/load and search."""  # Головний клас-сховище індексу.

    def __init__(self, chunks: Optional[List[Chunk]] = None) -> None:
        """Init with chunks and fresh tokenizer plus retriever."""
        # Зберігаємо список об'єктів Chunk (якщо передано None — ініціалізуємо порожнім списком).
        self.chunks: List[Chunk] = list(chunks) if chunks else []
        # Створюємо персональний екземпляр токенізатора.
        self.tokenizer = build_tokenizer()
        # Створюємо об'єкт BM25-ретривера.
        self.retriever: Any = bm25s.BM25()

    def build(self) -> None:
        """Index chunk texts enriched with file paths."""  # Побудова зваженого зворотного індексу BM25.
        if not self.chunks:
            return  # Якщо чанків немає — виходимо.
            
        # КЛЮЧОВИЙ ПРИЙОМ ДЛЯ RECALL@K:
        # До тексту кожного чанка на початку приєднується шлях файлу: f"{c.file_path} {c.text}".
        # Це дозволяє системі знаходити файл, навіть якщо запит шукає його за назвою (наприклад, "openai_server").
        texts = [f"{c.file_path} {c.text}" for c in self.chunks]
        
        # Токенізуємо весь корпус у форматі іменованих кортежів для прискорення bm25s.
        tokens = self.tokenizer.tokenize(texts, return_as="tuple")
        
        # Будуємо матриці TF-IDF / BM25 частот у пам'яті.
        self.retriever.index(tokens)

    def save(self, directory: str) -> None:
        """Persist chunks, meta and BM25 files under directory."""  # Збереження всіх структур на диск.
        root = Path(directory)
        idx_dir = root / "bm25_index"
        # Створюємо підпапку bm25_index (якщо її немає):
        idx_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Зберігаємо бінарні таблиці BM25 (розподіли частот слів).
        self.retriever.save(str(idx_dir), show_progress=False)
        
        # 2. Зберігаємо словник токенів (vocab), щоб не перебудовувати його наново.
        self.tokenizer.save_vocab(str(idx_dir))
        
        # 3. Зберігаємо повний масив чанків у chunks.json (перетворюємо dataclass Chunk на dict через __dict__).
        payload = [c.__dict__ for c in self.chunks]
        (root / "chunks.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        
        # 4. Зберігаємо meta.json із кількістю чанків та міткою часу created_at (потрібна для інвалідації кешу).
        meta = {
            "num_chunks": len(self.chunks),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str) -> "KnowledgeBase":
        """Restore index state from disk in milliseconds."""  # Швидке завантаження індексу без перерахунку корпуса.
        root = Path(directory)
        idx_dir = root / "bm25_index"
        
        # Створюємо новий порожній екземпляр KnowledgeBase.
        kb = cls([])
        
        # 1. Завантажуємо навчені матриці BM25 безпосередньо з файлів (load_corpus=False заощаджує пам'ять).
        kb.retriever = bm25s.BM25.load(str(idx_dir), load_corpus=False)
        
        # 2. Відновлюємо токенізатор та підвантажуємо його збережений словник слів.
        kb.tokenizer = build_tokenizer()
        kb.tokenizer.load_vocab(str(idx_dir))
        
        # 3. Зчитуємо chunks.json і реконструюємо список об'єктів Chunk із правильними оффсетами.
        raw: List[dict[str, Any]] = json.loads(
            (root / "chunks.json").read_text(encoding="utf-8")
        )
        kb.chunks = [Chunk(**item) for item in raw]
        
        return kb  # Повертаємо повністю готову до пошуку базу знань.

    def search_bm25(
        self, query: str, k: int = 10, file_type: str = "all"
    ) -> List[Tuple[Chunk, float]]:
        """Lexical search with optional file-type filtering."""  # Пошук за релевантністю BM25.
        # Захист від дегенеративних вхідних даних (порожній запит, k <= 0, порожня база):
        if not query or not query.strip() or k <= 0 or not self.chunks:
            return []
            
        # Пул кандидатів (pool): беремо з запасом (мінімум k * 3 або 50 документів),
        # тому що частина результатів може бути відкинута фільтром file_type (наприклад docs vs code).
        pool = min(len(self.chunks), max(k * 3, 50))
        
        # Токенізуємо вхідне запитання користувача:
        qtokens = self.tokenizer.tokenize([query], return_as="tuple")
        
        # Отримуємо ID найкращих документів і їхні бали релевантності BM25:
        ids, scores = self.retriever.retrieve(qtokens, k=pool)
        
        out: List[Tuple[Chunk, float]] = []
        # Проходимося по отриманих ID та балах:
        for idx, score in zip(ids[0].tolist(), scores[0].tolist()):
            chunk = self.chunks[int(idx)]  # Дістаємо чанк за індексом.
            
            # Якщо шлях до файлу чанка відповідає запитаній категорії (all / docs / code):
            if matches(chunk.file_path, file_type):
                out.append((chunk, float(score)))  # Додаємо у фінальний список.
                
            # Як тільки набрали необхідну кількість k результатів — зупиняємось:
            if len(out) >= k:
                break
                
        return out  # Повертаємо ранжований список пар (Chunk, score).


# Діагностичний блок для перевірки збереження/відновлення та пошуку:
if __name__ == "__main__":
    import shutil  # Для видалення тимчасової тестової папки після завершення тесту.

    print("=== [STEP 6 DIAGNOSTIC] Testing BM25s Indexing & Search ===")
    
    # 1. Створюємо три штучні чанки:
    demo = [
        Chunk("docs/lora.md", 100, 300, "load LoRA adapter weights dynamically"),
        Chunk("vllm/entrypoints/api.py", 50, 200, "entrypoint server code"),
        Chunk("docs/other.md", 0, 50, "unrelated cooking recipe"),
    ]
    
    # 2. Будуємо індекс і зберігаємо його в тимчасову директорію test_index:
    kb = KnowledgeBase(demo)
    kb.build()
    test_dir = "data/processed/test_index"
    kb.save(test_dir)
    print(f"[Index] Saved index and {len(demo)} chunks to {test_dir}/")
    
    # 3. Відновлюємо індекс із диска (тест методу load):
    kb2 = KnowledgeBase.load(test_dir)
    print(f"[OK] Index reloaded successfully. Total chunks: {len(kb2.chunks)}")
    
    # 4. Виконуємо пошуковий запит:
    q = "How to load LoRA adapter dynamically?"
    print(f"[Search Query]: '{q}'")
    hits = kb2.search_bm25(q, k=2)
    
    # Переконуємося, що на 1-му місці опинився саме документ docs/lora.md:
    assert hits and hits[0][0].file_path == "docs/lora.md"
    for rank, (ch, sc) in enumerate(hits, 1):
        print(f"  Rank #{rank} (Score: {sc:.2f}): "
              f"Path: {ch.file_path} [{ch.first_character_index}:"
              f"{ch.last_character_index}]")
              
    # Прибираємо за собою тимчасові тестові файли:
    shutil.rmtree(test_dir, ignore_errors=True)