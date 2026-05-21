import contextlib
import hashlib
import os
import random
import time
import wave
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv
try:
    from streamlit_mic_recorder import mic_recorder
except Exception:
    mic_recorder = None

from modules.feature_extraction import extract_features
from modules.fuzzy_evaluator import calculate_debate_score
from modules.openai_feedback import generate_counter_argument, generate_feedback
from modules.speech_to_text import transcribe_audio
from modules.utils import classify_debater, feature_labels, save_audio_to_temp_file

load_dotenv()

st.set_page_config(
    page_title="AI Debate Partner",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
    --bg: #0f172a;
    --bg-soft: #111827;
    --card: rgba(17, 24, 39, 0.72);
    --card-border: rgba(148, 163, 184, 0.18);
    --text: #e5e7eb;
    --muted: #94a3b8;
    --accent: #f59e0b;
    --accent-2: #38bdf8;
    --accent-3: #22c55e;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(56, 189, 248, 0.20), transparent 35%),
        radial-gradient(circle at top right, rgba(245, 158, 11, 0.18), transparent 30%),
        linear-gradient(180deg, #020617 0%, #0f172a 48%, #111827 100%);
    color: var(--text);
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.hero-card, .panel-card {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 22px;
    padding: 1.2rem 1.2rem 1rem 1.2rem;
    box-shadow: 0 20px 60px rgba(2, 6, 23, 0.35);
    backdrop-filter: blur(14px);
}

.hero-title {
    font-family: Georgia, "Times New Roman", serif;
    font-size: 2.35rem;
    font-weight: 700;
    line-height: 1.08;
    color: #ffffff;
    margin-bottom: 0.35rem;
}

.hero-subtitle {
    color: var(--muted);
    font-size: 1rem;
    line-height: 1.6;
}

.section-title {
    font-family: Georgia, "Times New Roman", serif;
    font-size: 1.2rem;
    color: #ffffff;
    margin-bottom: 0.6rem;
}

.small-note {
    color: var(--muted);
    font-size: 0.92rem;
}

.score-box {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 18px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.75rem;
}

.score-label {
    color: #cbd5e1;
    font-size: 0.88rem;
}

.score-value {
    color: #ffffff;
    font-size: 1.4rem;
    font-weight: 700;
}

[data-testid="stTextArea"] textarea {
    background: rgba(15, 23, 42, 0.75) !important;
    color: #f8fafc !important;
    border: 1px solid rgba(148, 163, 184, 0.20) !important;
    border-radius: 16px !important;
}

.mic-stage {
    min-height: 190px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    gap: 0.9rem;
}

iframe[title="streamlit_mic_recorder.streamlit_mic_recorder"] {
    width: 260px !important;
    min-height: 110px !important;
    display: block !important;
    margin: 0.25rem auto 1rem auto !important;
    border-radius: 18px !important;
    background:
        radial-gradient(circle at 42% 32%, rgba(56, 189, 248, 0.26), transparent 42%),
        rgba(15, 23, 42, 0.92);
    box-shadow: 0 24px 70px rgba(56, 189, 248, 0.22);
}

.mic-copy {
    color: var(--muted);
    max-width: 34rem;
    line-height: 1.6;
}

.recording-pill {
    color: #bfdbfe;
    background: rgba(37, 99, 235, 0.18);
    border: 1px solid rgba(96, 165, 250, 0.28);
    border-radius: 999px;
    padding: 0.55rem 0.9rem;
    font-size: 0.92rem;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def init_state() -> None:
    defaults = {
        "topic": "",
        "time_limit": 45,
        "side_choice": "Random",
        "ai_side": None,
        "user_side": None,
        "transcript": "",
        "audio_path": None,
        "audio_duration": 0.0,
        "feature_scores": {},
        "debate_score": None,
        "classification": "",
        "counter_argument": "",
        "feedback": "",
        "status_message": "",
        "processing_seconds": 0.0,
        "last_processed_audio_key": "",
        "auto_evaluate_recordings": True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def assign_sides() -> None:
    side_choice = st.session_state.get("side_choice", "Random")
    if side_choice in {"For", "Against"}:
        st.session_state.user_side = side_choice
        st.session_state.ai_side = "Against" if side_choice == "For" else "For"
    else:
        st.session_state.ai_side = random.choice(["For", "Against"])
        st.session_state.user_side = "Against" if st.session_state.ai_side == "For" else "For"

    st.session_state.status_message = (
        f"AI side assigned as {st.session_state.ai_side}. "
        f"You will debate from {st.session_state.user_side}."
    )


def handle_side_preference_change() -> None:
    assign_sides()
    clear_results(keep_sides=True)


def clear_results(*, keep_sides: bool = False) -> None:
    st.session_state.transcript = ""
    st.session_state.audio_path = None
    st.session_state.audio_duration = 0.0
    st.session_state.feature_scores = {}
    st.session_state.debate_score = None
    st.session_state.classification = ""
    st.session_state.counter_argument = ""
    st.session_state.feedback = ""
    st.session_state.processing_seconds = 0.0
    st.session_state.last_processed_audio_key = ""
    if keep_sides:
        st.session_state.status_message = (
            f"AI side assigned as {st.session_state.ai_side}. "
            f"You will debate from {st.session_state.user_side}."
        )
    else:
        st.session_state.status_message = ""


def get_audio_duration_seconds(audio_path: str) -> float:
    try:
        with contextlib.closing(wave.open(audio_path, "rb")) as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            if rate <= 0:
                return 0.0
            return round(frames / float(rate), 2)
    except Exception:
        return 0.0


def audio_source_to_bytes(audio_source: object) -> tuple[bytes | None, str, str, str]:
    """Normalize recorder/upload outputs into bytes, file name, MIME type, and a stable key."""
    audio_bytes = None
    audio_name = "response.wav"
    audio_format = "audio/wav"
    source_id = ""

    if isinstance(audio_source, dict) and audio_source.get("bytes"):
        audio_bytes = audio_source.get("bytes")
        recorder_format = str(audio_source.get("format") or "wav").lower().strip(".")
        audio_name = f"mic_recording.{recorder_format}"
        audio_format = f"audio/{recorder_format}"
        source_id = str(audio_source.get("id") or "")
    elif hasattr(audio_source, "getvalue"):
        audio_bytes = audio_source.getvalue()
        audio_name = getattr(audio_source, "name", "response.wav")
    elif hasattr(audio_source, "read"):
        audio_bytes = audio_source.read()
        audio_name = getattr(audio_source, "name", "response.wav")
    elif isinstance(audio_source, bytes):
        audio_bytes = audio_source

    if audio_bytes:
        digest = hashlib.sha256(audio_bytes).hexdigest()[:16]
        source_key = f"{audio_name}:{source_id}:{digest}"
    else:
        source_key = ""

    return audio_bytes, audio_name, audio_format, source_key


def save_audio_source(audio_source: object) -> tuple[str, str]:
    audio_bytes, audio_name, _audio_format, source_key = audio_source_to_bytes(audio_source)
    if not audio_bytes:
        raise RuntimeError("The selected audio file is empty.")

    suffix = Path(audio_name).suffix or ".wav"
    audio_path = save_audio_to_temp_file(audio_bytes, suffix=suffix)
    st.session_state.audio_path = audio_path
    st.session_state.audio_duration = get_audio_duration_seconds(audio_path)
    return audio_path, source_key


def transcribe_current_audio(audio_source: object) -> None:
    topic = st.session_state.get("topic", "").strip()
    if not topic:
        st.error("Please enter a debate topic first.")
        return

    clear_results()
    started_at = time.perf_counter()

    try:
        with st.spinner("Saving audio and transcribing with OpenAI..."):
            audio_path, source_key = save_audio_source(audio_source)
            transcript = transcribe_audio(audio_path)

        if not transcript.strip():
            st.error("OpenAI returned an empty transcript.")
            return

        st.session_state.transcript = transcript.strip()
        st.session_state.processing_seconds = round(time.perf_counter() - started_at, 2)
        st.session_state.status_message = (
            f"Transcription completed in {st.session_state.processing_seconds:.2f} seconds."
        )
        st.session_state.last_processed_audio_key = source_key
    except RuntimeError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Unexpected processing failure: {exc}")


def evaluate_current_audio(audio_source: object, *, auto_started: bool = False) -> None:
    topic = st.session_state.get("topic", "").strip()
    if not topic:
        st.error("Please enter a debate topic first.")
        return

    clear_results()
    if st.session_state.ai_side is None or st.session_state.user_side is None:
        assign_sides()

    started_at = time.perf_counter()

    try:
        with st.spinner("Saving audio and preparing transcription..."):
            audio_path, source_key = save_audio_source(audio_source)

        if st.session_state.audio_duration and st.session_state.audio_duration > st.session_state.time_limit:
            st.warning(
                f"Your response lasted {st.session_state.audio_duration:.1f}s, "
                f"which exceeds the selected limit of {st.session_state.time_limit}s."
            )

        with st.spinner("Transcribing speech with OpenAI..."):
            transcript = transcribe_audio(audio_path)
        if not transcript.strip():
            st.error("OpenAI returned an empty transcript.")
            return

        st.session_state.transcript = transcript.strip()

        with st.spinner("Extracting debate features..."):
            st.session_state.feature_scores = extract_features(
                transcript=st.session_state.transcript,
                topic=topic,
                elapsed_seconds=st.session_state.audio_duration,
                time_limit_seconds=st.session_state.time_limit,
            )

        with st.spinner("Running fuzzy debate evaluation..."):
            st.session_state.debate_score = calculate_debate_score(st.session_state.feature_scores)
            st.session_state.classification = classify_debater(st.session_state.debate_score)

        with st.spinner("Generating AI counter-argument and feedback..."):
            st.session_state.counter_argument = generate_counter_argument(
                topic=topic,
                user_side=st.session_state.user_side,
                ai_side=st.session_state.ai_side,
                transcript=st.session_state.transcript,
            )
            st.session_state.feedback = generate_feedback(
                topic=topic,
                user_side=st.session_state.user_side,
                ai_side=st.session_state.ai_side,
                transcript=st.session_state.transcript,
                feature_scores=st.session_state.feature_scores,
                debate_score=st.session_state.debate_score,
                classification=st.session_state.classification,
            )

        st.session_state.processing_seconds = round(time.perf_counter() - started_at, 2)
        prefix = "Auto evaluation" if auto_started else "Evaluation"
        st.session_state.status_message = f"{prefix} completed in {st.session_state.processing_seconds:.2f} seconds."
        st.session_state.last_processed_audio_key = source_key
    except RuntimeError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Unexpected processing failure: {exc}")


def render_metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="score-box">
            <div class="score-label">{label}</div>
            <div class="score-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_feature_scores(scores: dict) -> None:
    if not scores:
        return

    labels = feature_labels()
    values = [scores.get(label, 0.0) for label in labels]
    colors = ["#38bdf8", "#f59e0b", "#22c55e", "#a78bfa", "#fb7185", "#60a5fa", "#f97316"]

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    bars = ax.bar(labels, values, color=colors[: len(labels)], edgecolor="#e2e8f0", linewidth=0.6)
    ax.set_ylim(0, 10)
    ax.set_ylabel("Score / 10", color="#e5e7eb")
    ax.set_title("Debate Evaluation Criteria", color="#ffffff", pad=12, fontweight="bold")
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    ax.tick_params(axis="x", colors="#cbd5e1", rotation=20)
    ax.tick_params(axis="y", colors="#cbd5e1")
    for spine in ax.spines.values():
        spine.set_color("#334155")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.15,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#f8fafc",
        )

    st.pyplot(fig, clear_figure=True)


init_state()

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">AI Debate Partner</div>
        <div class="hero-subtitle">
            A voice-based debate evaluation system using OpenAI speech-to-text, fuzzy logic, and AI-generated feedback.
            Speak on any topic, get scored, and review a counter-argument in real time.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Controls")
    st.caption("Set up the debate, record your answer, and get feedback.")
    st.text_input("Debate topic", key="topic", placeholder="Example: Social media improves education")
    st.radio(
        "Your side",
        ["Random", "For", "Against"],
        key="side_choice",
        horizontal=True,
        on_change=handle_side_preference_change,
    )
    st.slider("Time limit (seconds)", min_value=30, max_value=60, step=5, key="time_limit")
    st.button("Assign Sides", use_container_width=True, on_click=assign_sides)
    st.button("Reset Results", use_container_width=True, on_click=clear_results)
    st.toggle("Auto-evaluate after mic recording", key="auto_evaluate_recordings")

    st.markdown("---")
    if st.session_state.ai_side and st.session_state.user_side:
        render_metric_card("AI Side", st.session_state.ai_side)
        render_metric_card("Your Side", st.session_state.user_side)
    else:
        st.info("Click Assign Sides to start the debate.")

    api_key_present = bool(os.getenv("OPENAI_API_KEY"))
    ai_provider = os.getenv("AI_PROVIDER", "openai").strip().lower()
    stt_provider = os.getenv("STT_PROVIDER", ai_provider).strip().lower()
    st.write("Speech-to-text:", "Groq" if stt_provider == "groq" else "OpenAI")
    st.write("Feedback provider:", "Groq" if ai_provider == "groq" else "OpenAI")
    if ai_provider == "groq" or stt_provider == "groq":
        st.write("Groq API key:", "Configured" if os.getenv("GROQ_API_KEY") else "Missing")
        if stt_provider != "groq":
            st.write("OpenAI STT key:", "Configured" if api_key_present else "Missing")
    else:
        st.write("OpenAI API key:", "Configured" if api_key_present else "Missing")

left_col, right_col = st.columns([1.12, 0.88], gap="large")

with left_col:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="mic-stage">
            <div class="section-title">Tap the mic to begin</div>
            <div class="recording-pill">Target response time: {st.session_state.time_limit} seconds</div>
            <div class="mic-copy">
                Your browser will ask for microphone permission the first time.
                Click the mic, speak for the selected time, then click stop.
                The transcript and evaluation will run automatically.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mic_audio = None
    if mic_recorder is not None:
        mic_audio = mic_recorder(
            start_prompt="🎙️ Start Recording",
            stop_prompt="■ Stop Recording",
            just_once=False,
            use_container_width=True,
            format="wav",
            key="debate_mic_recorder",
        )
    else:
        st.warning("Mic recorder component is unavailable. Please install dependencies and refresh.")

    audio_source = None
    if isinstance(mic_audio, dict) and mic_audio.get("bytes"):
        audio_source = mic_audio

    audio_key = ""
    if audio_source is not None:
        _audio_bytes, _audio_name, _audio_format, audio_key = audio_source_to_bytes(audio_source)

    fresh_recording = bool(audio_key) and audio_key != st.session_state.last_processed_audio_key
    if fresh_recording:
        if st.session_state.auto_evaluate_recordings:
            st.info("Recording received. Transcribing and evaluating now...")
            evaluate_current_audio(audio_source, auto_started=True)
        else:
            st.info("Recording received. Transcribing now...")
            transcribe_current_audio(audio_source)

    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.status_message:
        st.success(st.session_state.status_message)

    if st.session_state.processing_seconds:
        render_metric_card("Evaluation Time", f"{st.session_state.processing_seconds:.2f}s")

    if st.session_state.feature_scores:
        st.markdown('<div class="section-title">Extracted Feature Scores</div>', unsafe_allow_html=True)
        score_rows = [
            ("Logical Strength", st.session_state.feature_scores.get("Logical Strength", 0.0)),
            ("Evidence Usage", st.session_state.feature_scores.get("Evidence Usage", 0.0)),
            ("Clarity", st.session_state.feature_scores.get("Clarity", 0.0)),
            ("Emotional Bias", st.session_state.feature_scores.get("Emotional Bias", 0.0)),
            ("Fallacy Level", st.session_state.feature_scores.get("Fallacy Level", 0.0)),
            ("Time Efficiency", st.session_state.feature_scores.get("Time Efficiency", 0.0)),
            ("Relevance", st.session_state.feature_scores.get("Relevance", 0.0)),
        ]
        for label, value in score_rows:
            render_metric_card(label, f"{value:.1f}/10")

    if st.session_state.debate_score is not None:
        st.markdown('<div class="section-title">Final Result</div>', unsafe_allow_html=True)
        render_metric_card("Final Fuzzy Debate Score", f"{st.session_state.debate_score:.1f}/10")
        render_metric_card("Classification", st.session_state.classification)

with right_col:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Transcript Preview</div>', unsafe_allow_html=True)
    if st.session_state.transcript:
        st.text_area("Your transcribed words", value=st.session_state.transcript, height=300)
    else:
        st.text_area(
            "Your transcribed words",
            value="Your speech transcript will appear here after evaluation.",
            height=300,
        )

    st.markdown('<div class="section-title">AI Counter-Argument</div>', unsafe_allow_html=True)
    st.text_area(
        "Counter-argument",
        value=st.session_state.counter_argument or "The AI counter-argument will appear here after evaluation.",
        height=240,
    )
    st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.feature_scores:
    st.markdown('<div class="panel-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
    plot_feature_scores(st.session_state.feature_scores)
    st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.feedback:
    st.markdown('<div class="panel-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Personalized Feedback</div>', unsafe_allow_html=True)
    st.write(st.session_state.feedback)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="panel-card" style="margin-top: 1rem;">
        <div class="section-title">Viva / Presentation Note</div>
        <div class="small-note">
            This project uses OpenAI API for speech-to-text conversion and natural language feedback generation.
            Fuzzy logic is used for debate evaluation because argument quality is not always binary. A response can be
            partially logical, partially evidence-based, or moderately clear. Therefore, fuzzy inference is suitable for
            handling uncertainty in debate performance evaluation.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
