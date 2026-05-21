from __future__ import annotations

import re
import tempfile
from pathlib import Path

try:
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.tokenize import word_tokenize
except Exception:  # pragma: no cover - optional dependency fallback
    nltk_stopwords = None
    word_tokenize = None

FEATURE_NAMES = [
    "Logical Strength",
    "Evidence Usage",
    "Clarity",
    "Emotional Bias",
    "Fallacy Level",
    "Time Efficiency",
    "Relevance",
]

FALLBACK_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "me",
    "might",
    "my",
    "not",
    "of",
    "on",
    "or",
    "our",
    "she",
    "so",
    "such",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "will",
    "with",
    "would",
    "you",
}

if nltk_stopwords is not None:
    try:
        STOPWORDS = set(nltk_stopwords.words("english")) | FALLBACK_STOPWORDS
    except Exception:
        STOPWORDS = FALLBACK_STOPWORDS
else:
    STOPWORDS = FALLBACK_STOPWORDS


def feature_labels() -> list[str]:
    return FEATURE_NAMES


def tokenize(text: str) -> list[str]:
    if not text:
        return []

    if word_tokenize is not None:
        try:
            tokens = [token.lower() for token in word_tokenize(text)]
            return [token for token in tokens if re.fullmatch(r"[a-z']+", token)]
        except Exception:
            pass

    return re.findall(r"[a-z']+", text.lower())


def content_tokens(text: str) -> list[str]:
    return [token for token in tokenize(text) if token not in STOPWORDS]


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", text.lower())).strip()


def clamp_score(value: float, lower: float = 0.0, upper: float = 10.0) -> float:
    return max(lower, min(upper, float(value)))


def classify_debater(score: float) -> str:
    if score <= 4.0:
        return "Beginner Debater"
    if score <= 7.0:
        return "Intermediate Debater"
    return "Advanced Debater"


def save_audio_to_temp_file(audio_bytes: bytes, suffix: str = ".wav") -> str:
    if not audio_bytes:
        raise ValueError("Audio file is empty.")

    if not suffix.startswith("."):
        suffix = f".{suffix}"

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(audio_bytes)
    temp_file.flush()
    temp_file.close()
    return temp_file.name


def topic_keywords(topic: str) -> set[str]:
    return set(content_tokens(topic))


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def sentence_count(text: str) -> int:
    if not text.strip():
        return 0
    parts = [segment.strip() for segment in re.split(r"[.!?]+", text) if segment.strip()]
    return max(1, len(parts))
