"""SHA-256 file hashing for incremental indexing (Bonus 3)."""  # Докстрінг модуля: пояснює, що файл реалізує хешування файлів для інкрементальної індексації (Bonus 3).

import hashlib  # Стандартний модуль Python для криптографічних алгоритмів (зокрема SHA-256).
from typing import Dict, List, Set, Tuple  # Анотації типів для статичного типувальника mypy (вимога 42).


def compute_hash(text: str) -> str:
    """Compute hex SHA-256 of file content.

    Args:
        text: File content string.

    Returns:
        64-char hex digest.
    """
    # 1. text.encode("utf-8", errors="replace"): перетворюємо рядок у байти UTF-8.
    #    errors="replace" гарантує, що недійсні символи не викличуть винятку UnicodeEncodeError.
    # 2. hashlib.sha256(...): обчислюємо SHA-256 хеш від цих байтів.
    # 3. .hexdigest(): отримуємо результат як 64-символьний шістнадцятковий рядок.
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def diff_hashes(
    old_hashes: Dict[str, str], current_hashes: Dict[str, str]
) -> Tuple[Set[str], Set[str]]:
    """Diff two path->hash maps.

    Args:
        old_hashes: Hashes from previous indexing run.
        current_hashes: Hashes from current scan.

    Returns:
        Tuple (changed_or_new, deleted) as sets of relative paths.
    """
    # Отримуємо множину всіх шляхів файлів із попереднього збереженого індексу (hashes.json).
    old_keys = set(old_hashes)
    
    # Отримуємо множину всіх шляхів файлів, знайдених під час поточного сканування диску.
    cur_keys = set(current_hashes)
    
    # Знаходимо файли, які потрібно нарізати на чанки заново (changed):
    # Файл вважається новим або зміненим, якщо його хеш зараз відрізняється від старого
    # (якщо файлу не було в старій мапі, old_hashes.get(p) поверне None, що теж не дорівнює новому хешу).
    changed = {p for p in cur_keys if current_hashes[p] != old_hashes.get(p)}
    
    # Знаходимо видалені файли (deleted):
    # Це шляхи, які були в старому індексі, але більше не існують на диску (різниця множин).
    deleted = old_keys - cur_keys
    
    # Повертаємо кортеж із двох множин: (змінені_або_нові, видалені).
    return changed, deleted


def hashes_for(
    files: List[Tuple[str, str]], file_hashes: Dict[str, str] | None = None
) -> Dict[str, str]:
    """Build path->hash map for collected files.

    Args:
        files: (relative_path, text) pairs.
        file_hashes: Optional dict to fill and return.

    Returns:
        Dict mapping relative path to SHA-256 hex.
    """
    # Якщо користувач передав існуючий словник — наповнюємо його, якщо ні (None) — створюємо новий порожній.
    out: Dict[str, str] = file_hashes if file_hashes is not None else {}
    
    # Ітеруємося по кожному зібраному файлу: кортеж містить відносний шлях та повний текст файлу.
    for path, text in files:
        # Обчислюємо SHA-256 від тексту і записуємо в словник: {шлях: sha256_хеш}.
        out[path] = compute_hash(text)
        
    # Повертаємо підсумкову мапу хешів для всіх поточних файлів репозиторію.
    return out