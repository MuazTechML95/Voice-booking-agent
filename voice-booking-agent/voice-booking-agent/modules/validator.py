"""
modules/validator.py
----------------------
Generic field validators for the appointment booking agent.

These functions are intentionally industry-agnostic: they validate
*shapes* of data (a name, a phone number, a date, a time) and never
reference any specific business domain. This keeps validation rules
reusable across clinics, salons, law firms, etc.

Every validator follows the same contract:
    validate_xxx(text: str, **kwargs) -> (is_valid: bool, cleaned_value)
"""

import re
from datetime import datetime, date, time as dtime

try:
    from dateutil import parser as date_parser
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "python-dateutil is required. Install it with: pip install python-dateutil"
    ) from exc


PHONE_RE = re.compile(r"[\d+\-\s()]{7,20}")


def is_empty(text: str) -> bool:
    """Treat None, '', and whitespace-only strings as empty input."""
    return text is None or str(text).strip() == ""


def is_cancel_intent(text: str) -> bool:
    """Detect a user wanting to abandon the current flow."""
    text = (text or "").strip().lower()
    cancel_words = ["cancel", "stop", "exit", "quit", "nevermind", "never mind", "forget it", "abort"]
    return any(word in text for word in cancel_words)


def is_affirmative(text: str) -> bool:
    text = (text or "").strip().lower()
    return text in {
        "yes", "yeah", "yep", "sure", "ok", "okay", "correct",
        "proceed", "confirm", "y", "right", "of course",
    } or text.startswith("yes")


def is_negative(text: str) -> bool:
    text = (text or "").strip().lower()
    return text in {"no", "nope", "nah", "n", "incorrect", "wrong"} or text.startswith("no")


def validate_name(text: str):
    """A valid name must contain at least one alphabetic word and not be pure digits."""
    if is_empty(text):
        return False, None
    cleaned = text.strip()
    if not re.search(r"[A-Za-z]{2,}", cleaned):
        return False, None
    if re.fullmatch(r"[\d\s]+", cleaned):
        return False, None
    return True, " ".join(w.capitalize() for w in cleaned.split())


def validate_phone(text: str):
    """Accepts digits, spaces, '+', '-', and parentheses; requires 7-15 digits."""
    if is_empty(text):
        return False, None
    match = PHONE_RE.search(text)
    if not match:
        return False, None
    digits = re.sub(r"\D", "", match.group())
    if 7 <= len(digits) <= 15:
        return True, match.group().strip()
    return False, None


def validate_date(text: str, allow_past: bool = False):
    """Parse a natural-language date into ISO format (YYYY-MM-DD)."""
    if is_empty(text):
        return False, None
    try:
        parsed = date_parser.parse(text, fuzzy=True, default=datetime.now())
        parsed_date = parsed.date()
    except (ValueError, OverflowError):
        return False, None

    if not allow_past and parsed_date < date.today():
        return False, None
    return True, parsed_date.isoformat()


def validate_time(text: str, working_hours: dict = None):
    """Parse a natural-language time into 24h HH:MM format, optionally bounded by working hours."""
    if is_empty(text):
        return False, None
    try:
        # Fixed default (midnight) so unmentioned components (e.g. minutes
        # in "2 PM") default to 0 instead of leaking the current real time.
        zero_default = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        parsed = date_parser.parse(text, fuzzy=True, default=zero_default)
        parsed_time = parsed.time()
    except (ValueError, OverflowError):
        return False, None

    if working_hours:
        try:
            start = dtime.fromisoformat(working_hours["start"])
            end = dtime.fromisoformat(working_hours["end"])
        except (KeyError, ValueError):
            start = end = None
        if start and end and not (start <= parsed_time <= end):
            return False, None

    return True, parsed_time.strftime("%H:%M")


def validate_text(text: str):
    """Generic free-text validator (used for 'purpose' and similar fields)."""
    if is_empty(text):
        return False, None
    return True, text.strip()


def validate_category(text: str, categories: list):
    """Match free text against the configured category list, by substring or numeric index."""
    if is_empty(text) or not categories:
        return False, None
    text_low = text.strip().lower()
    for cat in categories:
        if text_low in cat.lower() or cat.lower() in text_low:
            return True, cat
    if text_low.isdigit():
        idx = int(text_low) - 1
        if 0 <= idx < len(categories):
            return True, categories[idx]
    return False, None


# Lookup table used by the conversation manager to pick the right
# validator for each configured question "type", without any
# if/elif chain scattered across the codebase.
VALIDATORS = {
    "name": validate_name,
    "phone": validate_phone,
    "date": validate_date,
    "time": validate_time,
    "text": validate_text,
}
