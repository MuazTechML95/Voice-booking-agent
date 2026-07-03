"""
modules/text_to_speech.py
----------------------------
Text-to-Speech wrapper built around pyttsx3.

pyttsx3 normally speaks directly out of the server's audio device,
which is meaningless in a web app (the person using the browser is
not sitting at the server). So instead this module renders speech to
a temporary .wav file using pyttsx3's `save_to_file`, and the
Streamlit UI plays that file back in the browser with `st.audio()`.

If pyttsx3 fails for any reason (no TTS engine available on the host,
e.g. some Linux containers without espeak installed), the app
degrades gracefully to text-only responses instead of crashing.
"""

import os
import tempfile
import uuid

from modules.utils import get_logger

logger = get_logger(__name__)


def synthesize_speech(text: str):
    """
    Convert text to speech and return the path to a .wav file the
    Streamlit UI can play, or None if speech synthesis isn't
    available / fails on this machine.
    """
    if not text or not text.strip():
        return None

    try:
        import pyttsx3
    except ImportError:
        logger.warning("pyttsx3 not installed; voice output disabled.")
        return None

    output_path = os.path.join(tempfile.gettempdir(), f"tts_{uuid.uuid4().hex}.wav")

    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)
        engine.save_to_file(text, output_path)
        engine.runAndWait()
        engine.stop()

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        return None

    except Exception as exc:  # noqa: BLE001
        logger.warning("Text-to-speech synthesis failed: %s", exc)
        return None
