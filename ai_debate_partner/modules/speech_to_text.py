from __future__ import annotations

import os
from pathlib import Path

from openai import OpenAI


def _stt_provider() -> str:
    return os.getenv("STT_PROVIDER", os.getenv("AI_PROVIDER", "openai")).strip().lower()


def _get_client() -> OpenAI:
    provider = _stt_provider()
    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GROQ_API_KEY. Add it to your environment or .env file.")
        return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY. Add it to your environment or .env file.")
    return OpenAI(api_key=api_key)


def transcribe_audio(audio_path: str, model: str | None = None) -> str:
    """Send a local audio file to OpenAI speech-to-text and return the transcript."""
    path = Path(audio_path)
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError("The audio file is missing or empty.")

    client = _get_client()
    if model:
        transcription_model = model
    elif _stt_provider() == "groq":
        transcription_model = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")
    else:
        transcription_model = os.getenv("OPENAI_STT_MODEL", "whisper-1")

    try:
        with path.open("rb") as audio_file:
            response = client.audio.transcriptions.create(
                model=transcription_model,
                file=audio_file,
            )
        transcript = getattr(response, "text", "")
        return transcript.strip()
    except Exception as exc:
        provider_name = "Groq" if _stt_provider() == "groq" else "OpenAI"
        raise RuntimeError(f"{provider_name} speech-to-text request failed: {exc}") from exc
