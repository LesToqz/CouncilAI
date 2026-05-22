from __future__ import annotations

from itertools import combinations


def _jaccard_similarity(left: str, right: str) -> float:
    left_words = {word.lower() for word in left.split() if word.strip()}
    right_words = {word.lower() for word in right.split() if word.strip()}
    if not left_words and not right_words:
        return 1.0
    if not left_words or not right_words:
        return 0.0
    return len(left_words & right_words) / len(left_words | right_words)


def calculate_similarity(texts: list[str]) -> float:
    """Return average similarity between texts.

    The MVP does not use this for early stopping. It is intentionally lightweight
    and falls back to Jaccard similarity if sklearn is unavailable.
    """
    cleaned = [text.strip() for text in texts if text and text.strip()]
    if len(cleaned) < 2:
        return 1.0

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectors = TfidfVectorizer().fit_transform(cleaned)
        matrix = cosine_similarity(vectors)
        scores = [matrix[i][j] for i, j in combinations(range(len(cleaned)), 2)]
        return float(sum(scores) / len(scores))
    except Exception:
        scores = [_jaccard_similarity(left, right) for left, right in combinations(cleaned, 2)]
        return sum(scores) / len(scores)
