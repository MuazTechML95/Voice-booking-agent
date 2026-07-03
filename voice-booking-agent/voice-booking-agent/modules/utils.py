"""
modules/utils.py
-----------------
Small shared helper functions used across the project:
- Loading the generic config.json
- Logging setup
- Generic string formatting helper used by the conversation manager

Keeping these in one place avoids duplicating boilerplate in every
module and keeps the rest of the codebase free of business-specific
logic, per the project requirement that nothing here should be
hardcoded for a particular industry.
"""

import json
import logging
import os
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.json")
DB_DIR = os.path.join(BASE_DIR, "database")
DB_PATH = os.path.join(DB_DIR, "appointments.db")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with consistent formatting."""
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_config(config_path: str = CONFIG_PATH) -> dict:
    """
    Load the generic configuration file (config/config.json).

    Raises a clear, user-friendly error if the file is missing or
    contains invalid JSON, instead of letting a raw traceback bubble
    up to the Streamlit UI.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found at '{config_path}'. "
            "Make sure config/config.json exists."
        )
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in '{config_path}': {exc}") from exc


def get_categories_for_business(config: dict, business_type: str) -> list:
    """
    Return the configured appointment categories for the selected
    business type, falling back to the generic 'default' list if the
    business type has no specific categories configured. This is what
    keeps the app generic/configurable across industries.
    """
    categories_map = config.get("appointment_categories", {})
    return categories_map.get(business_type) or categories_map.get("default", [])


def format_template(template: str, **data) -> str:
    """
    Safely format a message template defined in config.json using the
    given keyword data. Missing keys are ignored instead of raising,
    so a slightly mismatched template never crashes the conversation.
    """
    try:
        return template.format(**data)
    except (KeyError, IndexError):
        return template


def now_iso() -> str:
    """Current timestamp in ISO 8601 format, used for created_at."""
    return datetime.now().isoformat(timespec="seconds")


def ensure_dirs():
    """Create folders the app depends on if they don't already exist."""
    os.makedirs(DB_DIR, exist_ok=True)
