"""
modules/speech_to_text.py
----------------------------
Speech-to-Text wrapper built around Whisper.

Two backends are supported, chosen automatically in this order:
1. Local Whisper model (the `openai-whisper` pip package) — runs
   fully offline once the model weights are downloaded.
2. Groq's hosted Whisper API (`whisper-large-v3`) — free, very fast.
   Used when GROQ_API_KEY is configured.

Either way, the rest of the app only ever calls `transcribe_audio()`
and gets back a plain (success, text_or_error) tuple — it never needs
to know which backend handled the request.
"""
import os
import tempfile
from modules.utils import get_logger

logger = get_logger(__name__)

_local_model = None  # lazily loaded, cached for the lifetime of the process


def _get_local_model():
    """Lazily import and load the local Whisper model (small, good speed/accuracy trade-off)."""
    global _local_model
    if _local_model is not None:
        return _local_model
    try:
        import whisper  # openai-whisper package
    except ImportError:
        return None
    try:
        _local_model = whisper.load_model("base")
        logger.info("Loaded local Whisper 'base' model.")
        return _local_model
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load local Whisper model: %s", exc)
        return None


def _transcribe_local(audio_path: str):
    model = _get_local_model()
    if model is None:
        return False, None
    try:
        result = model.transcribe(audio_path)
        text = (result.get("text") or "").strip()
        return True, text
    except Exception as exc:  # noqa: BLE001
        logger.warning("Local Whisper transcription failed: %s", exc)
        return False, None


def _transcribe_via_groq_api(audio_path: str):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return False, None
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_file,
            )
        return True, (transcript.text or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Groq Whisper API transcription failed: %s", exc)
        return False, None


def transcribe_audio(audio_bytes: bytes, suffix: str = ".wav"):
    """
    Transcribe raw audio bytes (e.g. from Streamlit's microphone
    recorder widget) to text.

    Returns (success: bool, text_or_error_message: str).
    """
    if not audio_bytes:
        return False, "No audio was recorded. Please try again."

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name

        # Try local Whisper first (fully offline, if installed)
        ok, text = _transcribe_local(tmp_path)

        # Fall back to Groq (free, fast, hosted)
        if not ok:
            ok, text = _transcribe_via_groq_api(tmp_path)

        if not ok or not text:
            return False, (
                "Speech recognition failed. Please check your microphone, "
                "internet connection / API key, or try typing your response instead."
            )
        return True, text

    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error during transcription: %s", exc)
        return False, "Speech recognition failed due to an unexpected error."
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass