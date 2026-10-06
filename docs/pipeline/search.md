1. Команда search <query> --k <int>

Ця команда виконує інтерактивний пошук джерел для одного текстового запиту та
виводить знайдений список джерел у термінал як JSON.

```bash
uv run python -m src search "How to configure OpenAI server?" --k 5
```

Задіяні файли:

1.  src/__main__.py — точка входу модуля python -m src, передає керування в
    fire.Fire(CLI).
2.  src/cli.py — парсить аргументи query, k, а також необов'язкові
    index_directory та file_type. Викликає метод run_search(...).
3.  src/commands/search_cmd.py — функція run_search:
      - Перевіряє вхідні дані на порожнечу та валідність.
      - Викликає завантаження індексу.
      - Ініціалізує дисковий кеш із перевіркою відбитка індексу.
      - Запускає кешований пошук search_cached(...).
      - Серіалізує список MinimalSource через .model_dump() і друкує JSON у
        stdout.
4.  src/retrieval.py:
      - load_kb(...): безпечно підвантажує базу знань з диску.
      - fingerprint(...): читає дату створення індексу з meta.json для валідації
        кешу.
      - search_cached(...): перевіряє, чи запит є в кеші; якщо немає — запускає
        kb.search_bm25(...) та додає результат у кеш.
      - chunk_to_source(...): перетворює внутрішні об'єкти Chunk на
        Pydantic-моделі MinimalSource.
      - err(...): друкує помилки у sys.stderr у разі проблем.
5.  src/index.py — клас KnowledgeBase:
      - KnowledgeBase.load(...): зчитує chunks.json, бінарний індекс bm25_index/
        та відновлює словник токенізатора.
      - kb.search_bm25(...): токенізує запит користувача, рахує лексичні оцінки
        через bm25s і фільтрує за типом файлу.
6.  src/tokenizer.py — завантажує параметри токенізації та словник, розбиває
    запит на токени.
7.  src/file_filter.py — функція matches(...) перевіряє відповідність розширення
    знайденого файлу (docs, code або all).
8.  src/chunk_types.py — надає визначення dataclass Chunk.
9.  src/cache.py — клас QueryCache та функція cache_key(...): перевіряє
    наявність результату в data/processed/query_cache.json і записує туди нові
    результати за SHA-256 ключем.
10. src/models.py — модель MinimalSource валідує структуру (file_path,
    first_character_index, last_character_index).
11. src/__init__.py та src/commands/__init__.py — файли пакетів.