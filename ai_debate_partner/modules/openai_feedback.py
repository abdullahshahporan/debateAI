from __future__ import annotations

import json
import os

from openai import OpenAI


def _provider() -> str:
    return os.getenv("AI_PROVIDER", "openai").strip().lower()


def _get_client() -> OpenAI:
    provider = _provider()
    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GROQ_API_KEY. Add it to your environment or .env file.")
        return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY. Add it to your environment or .env file.")
    return OpenAI(api_key=api_key)


def _model_name() -> str:
    if _provider() == "groq":
        return os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _call_openai(prompt: str, system_prompt: str) -> str:
    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=_model_name(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )
        message = response.choices[0].message
        return (message.content or "").strip()
    except Exception as exc:
        provider_name = "Groq" if _provider() == "groq" else "OpenAI"
        raise RuntimeError(f"{provider_name} generation failed: {exc}") from exc


def generate_counter_argument(topic: str, user_side: str, ai_side: str, transcript: str) -> str:
    prompt = (
        "Generate a concise and logical counter-argument against the following user argument. "
        "Keep it suitable for academic debate practice. "
        f"Topic: {topic}. User side: {user_side}. AI side: {ai_side}. User argument: {transcript}."
    )
    system_prompt = "You are a skilled debate opponent. Respond with a short, sharp, evidence-aware counter-argument."
    return _call_openai(prompt, system_prompt)


def generate_feedback(
    topic: str,
    user_side: str,
    ai_side: str,
    transcript: str,
    feature_scores: dict[str, float],
    debate_score: float,
    classification: str,
) -> str:
    score_summary = json.dumps(feature_scores, indent=2)
    prompt = (
        "Provide personalized feedback for a debate practice session. Include strengths, weaknesses, "
        "improvement suggestions, and one short example of a stronger argument. "
        f"Topic: {topic}. User side: {user_side}. AI side: {ai_side}.\n"
        f"Transcript: {transcript}\n"
        f"Feature scores: {score_summary}\n"
        f"Final fuzzy score: {debate_score}/10. Classification: {classification}."
    )
    system_prompt = "You are a debate coach. Give practical, structured, encouraging but direct feedback."
    return _call_openai(prompt, system_prompt)
