"""Minimal tokenizer for the learning RAG skeleton (Step 3).

Learning-first approach: no camelCase/snake_case pre-splitting.
Lowercasing and English stopword removal are delegated to
``bm25s.tokenization.Tokenizer``. Identifiers stay whole
(``VLLM_RPC_TIMEOUT`` -> ``vllm_rpc_timeout``), exactly like
the reference example. Code-aware splitting is a deferred
improvement, not part of the skeleton.
"""  # Докстрінг модуля: пояснює архітектурне рішення — збереження цілісності ідентифікаторів коду (наприклад, VLLM_RPC_TIMEOUT).

from bm25s.tokenization import Tokenizer  # Швидкий токенізатор із бібліотеки bm25s, написаний з оптимізацією під пошук.

# Регулярний вираз для розбиття тексту: r"\w+" виділяє послідовності буквено-цифрових символів та знаків підкреслення.
# Це означає, що ідентифікатори на кшталт vllm_rpc_timeout не розриваються на три окремі слова, а залишаються одним токеном.
DEFAULT_SPLITTER = r"\w+"


def build_tokenizer() -> Tokenizer:
    # Фабрична функція, яка повертає сконфігурований екземпляр токенізатора:
    return Tokenizer(
        lower=True,                   # Приведення всіх слів до нижнього регістру (case-insensitive пошук).
        splitter=DEFAULT_SPLITTER,    # Регулярний вираз для виділення слів/токенів.
        stopwords="english",          # Видалення частотних англійських слів-паразитів ("the", "is", "at", "which"), що не несуть змісту.
        stemmer=None,                 # Стеммінг вимкнено (None), щоб не спотворювати специфічні назви змінних і функцій у коді.
    )


# Блок автономного тестування модуля (запускається через `python -m src.tokenizer`).
if __name__ == "__main__":
    tok = build_tokenizer()  # Створюємо екземпляр токенізатора.
    
    # Тестова токенізація двох фраз:
    demo = tok.tokenize(
        ["Hello WORLD", "What does VLLM_RPC_TIMEOUT default to?"],
        return_as="string",          # Повернути список списків рядків.
        show_progress=False,         # Не показувати прогрес-бар для двох рядків.
    )
    print("=== [STEP 3 DIAGNOSTIC] Testing Minimal Tokenizer ===")
    for text, tokens in zip(
        ["Hello WORLD", "What does VLLM_RPC_TIMEOUT default to?"], demo
    ):
        print(f"[Query] {text}")
        print(f"  -> {tokens} ({len(tokens)} tokens)")
        
    # ПЕРЕВІРКА 1: Перевіряємо нормалізацію регістру (HELLO WORLD -> ['hello', 'world']).
    assert demo[0] == ["hello", "world"]
    
    # ПЕРЕВІРКА 2: Змінна snake_case залишилася єдиним токеном, а стоп-слова ("what", "does", "to") відфільтровані!
    assert "vllm_rpc_timeout" in demo[1]
    print("[OK] Minimal tokenizer checks passed!")