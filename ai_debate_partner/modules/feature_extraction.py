from __future__ import annotations

import math
import re
from typing import Any

from textblob import TextBlob

from .fallacy_detector import detect_fallacies
from .utils import clamp_score, content_tokens, safe_ratio, sentence_count, tokenize

LOGICAL_CONNECTORS = [
    "because",
    "therefore",
    "since",
    "thus",
    "hence",
    "as a result",
    "consequently",
]

EVIDENCE_KEYWORDS = [
    "research",
    "data",
    "statistics",
    "survey",
    "study",
    "according to",
    "report",
    "example",
    "evidence",
    "analysis",
    "findings",
    "experiment",
]

AGGRESSION_WORDS = [
    "hate",
    "stupid",
    "idiot",
    "useless",
    "trash",
    "shut up",
    "ridiculous",
    "nonsense",
    "pathetic",
]


def _count_phrase_hits(text: str, phrases: list[str]) -> int:
    lowered = text.lower()
    count = 0
    for phrase in phrases:
        count += len(re.findall(rf"\b{re.escape(phrase)}\b", lowered))
    return count


def _logical_strength(text: str) -> float:
    words = tokenize(text)
    sentence_total = sentence_count(text)
    connector_hits = _count_phrase_hits(text, LOGICAL_CONNECTORS)

    if not words:
        return 0.0

    score = connector_hits * 1.6
    score += 1.5 if sentence_total >= 2 else 0.0
    score += 1.2 if any(keyword in text.lower() for keyword in ["claim", "reason", "conclusion"]) else 0.0

    if len(words) < 12:
        score -= 3.0
    elif len(words) < 25:
        score -= 1.0
    elif len(words) >= 80:
        score += 1.5

    if sentence_total == 1 and len(words) < 30:
        score -= 1.2

    return clamp_score(score)


def _evidence_usage(text: str) -> float:
    evidence_hits = _count_phrase_hits(text, EVIDENCE_KEYWORDS)
    numeric_cues = len(re.findall(r"\b\d+(?:\.\d+)?%?\b", text))
    example_cues = len(re.findall(r"\b(for example|for instance|such as|according to)\b", text.lower()))

    score = evidence_hits * 1.8
    score += numeric_cues * 0.8
    score += example_cues * 1.2

    if any(phrase in text.lower() for phrase in ["study shows", "report says", "data shows", "research indicates"]):
        score += 1.5

    return clamp_score(score)


def _clarity(text: str) -> float:
    words = tokenize(text)
    sentences = [segment.strip() for segment in re.split(r"[.!?]+", text) if segment.strip()]
    if not words:
        return 0.0

    average_sentence_length = safe_ratio(len(words), max(1, len(sentences)))
    repeated_tokens = len(words) - len(set(words))
    repeated_ratio = safe_ratio(repeated_tokens, max(1, len(words)))
    incomplete_penalty = 2.0 if text.strip() and text.strip()[-1] not in ".!?" else 0.0

    if 8 <= average_sentence_length <= 22:
        structure_score = 4.0
    elif 5 <= average_sentence_length < 8 or 22 < average_sentence_length <= 30:
        structure_score = 2.8
    else:
        structure_score = 1.4

    score = 10.0
    score -= abs(average_sentence_length - 16) * 0.15
    score -= repeated_ratio * 4.0
    score -= incomplete_penalty
    score += structure_score

    if len(words) < 10:
        score -= 3.0
    if len(sentences) == 1 and len(words) > 45:
        score -= 1.0

    return clamp_score(score)


def _emotional_bias(text: str) -> float:
    if not text:
        return 0.0

    sentiment = TextBlob(text).sentiment.polarity
    exclamations = text.count("!")
    uppercase_words = sum(1 for word in re.findall(r"\b[A-Z]{3,}\b", text))
    aggression_hits = _count_phrase_hits(text, AGGRESSION_WORDS)

    score = abs(sentiment) * 5.0
    score += exclamations * 1.0
    score += uppercase_words * 0.8
    score += aggression_hits * 1.8

    return clamp_score(score)


def _time_efficiency(text: str, elapsed_seconds: float, time_limit_seconds: int) -> float:
    words = tokenize(text)
    word_count = len(words)
    if word_count == 0:
        return 0.0

    if elapsed_seconds <= 0:
        return clamp_score(min(10.0, word_count / 10.0))

    limit = max(1, time_limit_seconds)
    ratio = elapsed_seconds / limit

    if ratio <= 1.0:
        closeness = 1.0 - abs(ratio - 0.85) / 0.85
        base = max(0.0, closeness * 8.0)
        length_bonus = min(2.0, word_count / 30.0)
        score = base + length_bonus
    else:
        overtime_penalty = (ratio - 1.0) * 6.0
        score = 8.0 - overtime_penalty

    if word_count < 12:
        score -= 3.0
    elif word_count < 25:
        score -= 1.0
    else:
        score += 1.0

    return clamp_score(score)


def _relevance(text: str, topic: str) -> float:
    topic_terms = set(content_tokens(topic))
    response_terms = set(content_tokens(text))
    if not topic_terms or not response_terms:
        return 0.0

    overlap = topic_terms.intersection(response_terms)
    jaccard = safe_ratio(len(overlap), len(topic_terms.union(response_terms)))
    phrase_bonus = 0.0

    topic_lower = topic.lower()
    text_lower = text.lower()
    for term in list(topic_terms)[:5]:
        if term and term in text_lower and term in topic_lower:
            phrase_bonus += 0.6

    return clamp_score(jaccard * 10.0 + phrase_bonus)


def extract_features(
    transcript: str,
    topic: str,
    elapsed_seconds: float,
    time_limit_seconds: int,
) -> dict[str, float]:
    """Extract the debate features used by the fuzzy evaluator."""
    if not transcript or not transcript.strip():
        raise ValueError("Transcript is empty.")

    fallacy_result = detect_fallacies(transcript)

    features = {
        "Logical Strength": _logical_strength(transcript),
        "Evidence Usage": _evidence_usage(transcript),
        "Clarity": _clarity(transcript),
        "Emotional Bias": _emotional_bias(transcript),
        "Fallacy Level": fallacy_result["score"],
        "Time Efficiency": _time_efficiency(transcript, elapsed_seconds, time_limit_seconds),
        "Relevance": _relevance(transcript, topic),
    }

    return {name: round(clamp_score(score), 1) for name, score in features.items()}
