from __future__ import annotations

import re
from typing import Any

from .utils import clamp_score

FALLACY_PATTERNS = {
    "ad_hominem": [
        r"\byou (are|re) (stupid|ignorant|lazy|lying|wrong|pathetic|useless)\b",
        r"\bidiot\b",
        r"\bfool\b",
        r"\bshut up\b",
    ],
    "overgeneralization": [
        r"\beveryone\b",
        r"\balways\b",
        r"\bnever\b",
        r"\ball people\b",
        r"\bnobody\b",
        r"\bevery single\b",
    ],
    "false_dilemma": [
        r"\beither\b.*\bor\b",
        r"\bonly two choices\b",
        r"\bthere are only two options\b",
    ],
    "slippery_slope": [
        r"\bif .* then .* (collapse|disaster|chaos|ruin|end)\b",
        r"\bwill lead to .* (disaster|chaos|collapse|ruin)\b",
        r"\bone small step .* (ruin|disaster|collapse)\b",
    ],
}


def detect_fallacies(text: str) -> dict[str, Any]:
    """Return a basic fallacy score and the detected patterns."""
    if not text:
        return {"score": 0.0, "hits": []}

    lowered = text.lower()
    hits: list[str] = []
    score = 0.0

    for fallacy_name, patterns in FALLACY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, lowered):
                hits.append(fallacy_name)
                score += 2.5
                break

    if len(re.findall(r"\b(yeah right|obviously|clearly|everyone knows)\b", lowered)) > 0:
        hits.append("dismissive_language")
        score += 1.5

    return {"score": clamp_score(score), "hits": hits}
