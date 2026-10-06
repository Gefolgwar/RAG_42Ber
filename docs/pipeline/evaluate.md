Команда evaluate --student_search_results_path <path> --dataset_path <path>

Це ваша власна реалізація розрахунку Recall@k, призначена для самоперевірки (на
відміну від зовнішнього бінарника moulinette).

```bash
uv run python -m src evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

Задіяні файли:

1.  src/__main__.py — стандартна точка входу.
2.  src/cli.py — метод CLI.evaluate(...) викликає функцію run_evaluate(...).
3.  src/commands/eval_cmd.py — функція run_evaluate:
      - Зчитує з диска обидва JSON-файли (файл результатів пошуку студента та
        файл еталонних відповідей із джерелами).
      - Зіставляє питання за їхніми question_id.
      - Для кожного питання запускає розрахунок метрики за значеннями
        k \in (1, 3, 5, 10).
      - Розраховує середнє арифметичне Recall для кожного k і виводить звіт у
        термінал із позначкою (PASS >= 80%) для Recall@5, якщо поріг пройдено.
4.  src/evaluator.py — містить математику розрахунку метрики згідно з правилами
    суб'єкта:
      - overlap_ratio(retrieved, truth): обчислює перекриття символів (IoU)
        відносно довжини еталонного відрізка (потрібно збіг file_path та
        перекриття \ge 5\%).
      - recall_at_k(...): перевіряє, яку частку еталонних джерел було успішно
        знайдено серед перших k результатів пошуку.
      - Константи THRESHOLD = 0.05 та KS = (1, 3, 5, 10).
5.  src/models.py — модель MinimalSource (використовується для валідації
    координат відрізків як у результатах студента, так і в еталоні).
6.  src/retrieval.py — функція err(...) для перехоплення помилок читання або
    некоректного JSON.
7.  src/__init__.py та src/commands/__init__.py — файли пакетів.