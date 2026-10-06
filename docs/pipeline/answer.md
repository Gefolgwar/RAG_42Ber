2. Команда answer <query> --k <int>

Ця команда об'єднує пошук і генерацію для одного запиту: спочатку знаходить
топ-k чанків через BM25, підставляє їх у промпт, генерує відповідь локальною
моделлю Qwen та виводить результат як JSON-об'єкт MinimalAnswer.

```bash
uv run python -m src answer "How to load LoRA adapter dynamically?" --k 5
```

Задіяні файли:

1.  src/__main__.py — точка входу, передає запит у CLI.
2.  src/cli.py — метод CLI.answer(...) перенаправляє виконання у функцію
    run_answer(...).
3.  src/commands/answer_cmd.py — функція run_answer:
      - Перевіряє валідність аргументів запиту.
      - Завантажує індекс через load_kb(...).
      - Виконує прямий пошук kb.search_bm25(...) для отримання об'єктів Chunk.
      - Завантажує генератор через get_generator() і викликає метод
        generate(query, chunks).
      - Формує фінальний об'єкт MinimalAnswer і виводить JSON у stdout через
        model_dump_json(indent=2).
4.  src/retrieval.py:
      - load_kb(...): завантажує індекс.
      - chunk_to_source(...): перетворює відібрані Chunk у моделі MinimalSource.
      - err(...): обробка винятків без аварійного завершення.
5.  src/index.py — KnowledgeBase.load(...) відновлює індекс та список чанків, а
    search_bm25(...) здійснює лексичний пошук і повертає топ-k чанків із
    текстом.
6.  src/tokenizer.py — токенізує запит користувача для лексичного пошуку BM25.
7.  src/file_filter.py — фільтрація документів за типом файлу всередині пошуку
    BM25.
8.  src/chunk_types.py — об'єкти Chunk, які містять і координати, і сам текст
    фрагментів.
9.  src/generator.py — підготовка та виклик моделі:
      - get_generator(): створює або повертає кешований екземпляр
        AnswerGenerator.
      - _ensure_loaded(): завантажує токенізатор і ваги моделі Qwen/Qwen3-0.6B.
      - format_context(chunks): форматує контекст у блоки [Source 1] ....
      - build_prompt(question, context): компонує системну інструкцію, контекст
        і питання.
      - generate(...): виконує інференс нейромережі та повертає згенерований
        текст.
10. src/models.py — моделі MinimalSource та MinimalAnswer для структуризації
    фінальної відповіді.
11. src/__init__.py та src/commands/__init__.py — файли пакетів.
