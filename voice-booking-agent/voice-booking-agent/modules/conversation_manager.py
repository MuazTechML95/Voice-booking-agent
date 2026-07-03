"""
modules/conversation_manager.py
---------------------------------
The heart of the agent: a generic, config-driven conversation state
machine. It contains NO business-specific logic — all wording,
categories, and questions come from config/config.json plus the
business_type chosen in the UI, so the exact same class works for a
medical clinic, a salon, a law firm, etc.

OpenAI's API is used for two NLP tasks where rule-based matching is
fragile:
    1. Greeting-stage intent detection ("book" vs "cancel" vs chit-chat)
    2. A light "did the user actually answer the question" sanity check

If no OPENAI_API_KEY is configured, or the API call fails for any
reason (network/quota/etc.), the agent transparently falls back to
simple keyword-based detection so the app keeps working end-to-end.

States:
    GREETING    -> waiting for the user to confirm they want to book
    COLLECTING  -> looping through the configured questions
    CONFIRMING  -> reading back collected info, waiting for yes/no
    DONE        -> appointment booked
    CANCELLED   -> user cancelled
"""

import os

from modules.validator import (
    VALIDATORS, is_cancel_intent, is_affirmative, is_negative,
    validate_category, is_empty,
)
from modules.knowledge_base import looks_like_knowledge_query, answer_question
from modules.utils import format_template, get_categories_for_business, get_logger

logger = get_logger(__name__)

# OpenAI is optional: the app must still run if the package or key is missing.
try:
    from openai import OpenAI
    _OPENAI_IMPORT_OK = True
except ImportError:
    _OPENAI_IMPORT_OK = False


def _detect_intent_via_openai(text: str) -> str:
    """
    Ask the OpenAI API to classify the user's free-text reply into a
    coarse intent. Returns one of: 'book', 'negative', 'unknown'.
    Returns 'unknown' on any error so the caller can fall back safely.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not _OPENAI_IMPORT_OK or not api_key:
        return "unknown"

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the user's message into exactly one word: "
                        "'book' if they want to book/schedule an appointment, "
                        "'negative' if they are declining/refusing, or "
                        "'unknown' if unclear. Reply with only that one word."
                    ),
                },
                {"role": "user", "content": text},
            ],
            max_tokens=5,
            temperature=0,
        )
        label = response.choices[0].message.content.strip().lower()
        if label in {"book", "negative", "unknown"}:
            return label
        return "unknown"
    except Exception as exc:  # noqa: BLE001 - any API error must not crash the app
        logger.warning("OpenAI intent detection failed, falling back to rules: %s", exc)
        return "unknown"


def _detect_intent_rule_based(text: str) -> str:
    """Keyword-based fallback intent detector (no external dependency)."""
    text_low = (text or "").strip().lower()
    booking_keywords = ["book", "appointment", "schedule", "meeting", "consult", "reserve", "slot", "visit"]
    if any(k in text_low for k in booking_keywords):
        return "book"
    if text_low in {"hi", "hello", "hey", "yes", "sure", "ok", "okay"}:
        return "book"
    return "unknown"


def detect_booking_intent(text: str) -> str:
    """Try OpenAI first, fall back to rules — see module docstring."""
    intent = _detect_intent_via_openai(text)
    if intent == "unknown":
        intent = _detect_intent_rule_based(text)
    return intent


class ConversationManager:
    STATE_GREETING = "GREETING"
    STATE_ASK_CATEGORY = "ASK_CATEGORY"
    STATE_COLLECTING = "COLLECTING"
    STATE_CONFIRMING = "CONFIRMING"
    STATE_DONE = "DONE"
    STATE_CANCELLED = "CANCELLED"

    def __init__(self, config: dict, business_type: str):
        self.config = config
        self.business_type = business_type
        self.categories = get_categories_for_business(config, business_type)
        self.reset()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def reset(self):
        self.state = self.STATE_GREETING
        self.questions = self.config.get("questions", [])
        self.q_index = 0
        self.answers = {"business_type": self.business_type}
        self.error = None

    def _fmt(self, key: str, **extra) -> str:
        template = self.config.get("messages", {}).get(key, "")
        data = {
            "business_name": self.config.get("business_name", "the business"),
            "business_type": self.business_type,
            "categories": ", ".join(self.categories),
        }
        data.update(self.answers)
        data.update(extra)
        return format_template(template, **data)

    def greeting_message(self) -> str:
        return self._fmt("greeting")

    def render_message(self, key: str, **extra) -> str:
        """Public helper so callers outside this class (e.g. the Streamlit
        UI) can render a templated message without touching internals."""
        return self._fmt(key, **extra)

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def handle_message(self, text: str) -> dict:
        """
        Process one user utterance.
        Returns: {reply, state, done, cancelled, appointment}
        """
        text = (text or "").strip()
        self.error = None

        if is_empty(text):
            return self._response(self._fmt("not_understood_message"))

        # Global cancel intent works from any non-terminal state.
        if self.state not in (self.STATE_DONE, self.STATE_CANCELLED) and is_cancel_intent(text):
            self.state = self.STATE_CANCELLED
            return self._response(self._fmt("cancel_message"))

        # FAQ / services / pricing questions can be asked at any point in
        # the conversation (RAG-backed). We answer them, then re-show
        # whatever the agent was waiting for, without losing booking
        # progress. While collecting a structured field, we use the
        # stricter check so a normal answer (e.g. "general checkup" as a
        # purpose) is never mistaken for a question.
        if self.state not in (self.STATE_DONE, self.STATE_CANCELLED):
            strict = self.state == self.STATE_COLLECTING
            if looks_like_knowledge_query(text, strict=strict):
                ok, answer = answer_question(self.config, self.business_type, text)
                pending = self._pending_prompt()
                combined = f"{answer}\n\n{pending}".strip() if pending else answer
                return self._response(combined)

        if self.state == self.STATE_GREETING:
            return self._handle_greeting(text)
        if self.state == self.STATE_ASK_CATEGORY:
            return self._handle_category(text)
        if self.state == self.STATE_COLLECTING:
            return self._handle_collecting(text)
        if self.state == self.STATE_CONFIRMING:
            return self._handle_confirming(text)

        # Terminal states: a new message restarts a fresh booking.
        self.reset()
        return self._response(self.greeting_message())

    # ------------------------------------------------------------------ #
    # State handlers
    # ------------------------------------------------------------------ #
    def _handle_greeting(self, text: str):
        intent = detect_booking_intent(text)

        if intent == "book" or is_affirmative(text):
            if self.categories:
                self.state = self.STATE_ASK_CATEGORY
                return self._response(self._fmt("ask_category_prompt"))
            self.state = self.STATE_COLLECTING
            return self._response(self._current_question_prompt())

        if intent == "negative" or is_negative(text):
            self.state = self.STATE_CANCELLED
            return self._response(self._fmt("goodbye_message"))

        return self._response(self._fmt("not_understood_message"))

    def _handle_category(self, text: str):
        ok, value = validate_category(text, self.categories)
        if not ok:
            return self._response(f"Please choose one of: {', '.join(self.categories)}.")
        self.answers["category"] = value
        self.state = self.STATE_COLLECTING
        return self._response(self._current_question_prompt())

    def _handle_collecting(self, text: str):
        question = self.questions[self.q_index]
        validator = VALIDATORS.get(question["type"], VALIDATORS["text"])

        if question["type"] == "time" and self.config.get("working_hours"):
            ok, value = validator(text, working_hours=self.config["working_hours"])
        else:
            ok, value = validator(text)

        if not ok:
            self.error = f"Invalid input for '{question['key']}'."
            return self._response(question["retry_prompt"])

        self.answers[question["key"]] = value
        self.q_index += 1

        if self.q_index >= len(self.questions):
            self.state = self.STATE_CONFIRMING
            return self._response(self._fmt("confirmation_template"))

        return self._response(self._current_question_prompt())

    def _handle_confirming(self, text: str):
        if is_affirmative(text):
            self.state = self.STATE_DONE
            return self._response("CONFIRMED", done=True, appointment=dict(self.answers))

        if is_negative(text):
            # Restart data collection, keep business_type and category.
            self.q_index = 0
            self.answers = {
                k: v for k, v in self.answers.items()
                if k in ("business_type", "category")
            }
            self.state = self.STATE_COLLECTING
            return self._response("No problem, let's go through the details again. " + self._current_question_prompt())

        return self._response("Please answer yes or no — should I proceed with the booking?")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _current_question_prompt(self) -> str:
        question = self.questions[self.q_index]
        data = {"business_type": self.business_type}
        data.update(self.answers)
        return format_template(question["prompt"], **data)

    def _pending_prompt(self) -> str:
        """Re-render whatever prompt the agent was waiting on, without
        changing state — used after answering a side-question via RAG."""
        if self.state == self.STATE_GREETING:
            return self.greeting_message()
        if self.state == self.STATE_ASK_CATEGORY:
            return self._fmt("ask_category_prompt")
        if self.state == self.STATE_COLLECTING:
            return self._current_question_prompt()
        if self.state == self.STATE_CONFIRMING:
            return self._fmt("confirmation_template")
        return ""

    def _response(self, reply: str, done: bool = False, appointment: dict = None) -> dict:
        return {
            "reply": reply,
            "state": self.state,
            "done": done,
            "cancelled": self.state == self.STATE_CANCELLED,
            "appointment": appointment,
            "error": self.error,
        }
