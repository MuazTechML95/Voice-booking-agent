"""
modules/database.py
---------------------
SQLite persistence layer for appointments.

Table schema (as required by the project spec):
    appointments
        id                  INTEGER PRIMARY KEY AUTOINCREMENT
        business_type       TEXT
        full_name           TEXT
        phone               TEXT
        appointment_date    TEXT
        appointment_time    TEXT
        purpose             TEXT
        created_at          TEXT

Two extra, non-breaking columns are included to support the optional
"category", reschedule, and cancel features without altering the
required columns above:
        category            TEXT
        status              TEXT  ('CONFIRMED' / 'CANCELLED')

The database file is created automatically the first time this
module is imported, so the app works on a fresh checkout with zero
manual setup.
"""

import sqlite3
from contextlib import contextmanager

from modules.utils import DB_PATH, ensure_dirs, now_iso, get_logger

logger = get_logger(__name__)


@contextmanager
def get_connection():
    """Context-managed SQLite connection so callers never forget to close it."""
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create the appointments table if it doesn't already exist."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS appointments (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                business_type     TEXT NOT NULL,
                full_name         TEXT NOT NULL,
                phone             TEXT NOT NULL,
                appointment_date  TEXT NOT NULL,
                appointment_time  TEXT NOT NULL,
                purpose           TEXT,
                category          TEXT,
                status            TEXT NOT NULL DEFAULT 'CONFIRMED',
                created_at        TEXT NOT NULL
            )
            """
        )
    logger.info("Database ready at %s", DB_PATH)


def insert_appointment(data: dict) -> int:
    """
    Insert a new appointment row and return its generated id.

    `data` is expected to contain: business_type, full_name, phone,
    appointment_date, appointment_time, purpose, category(optional).
    """
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO appointments
                (business_type, full_name, phone, appointment_date,
                 appointment_time, purpose, category, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data.get("business_type"),
                data.get("full_name"),
                data.get("phone"),
                data.get("appointment_date"),
                data.get("appointment_time"),
                data.get("purpose"),
                data.get("category"),
                "CONFIRMED",
                now_iso(),
            ),
        )
        return cursor.lastrowid


def update_appointment(appointment_id: int, appointment_date: str, appointment_time: str) -> bool:
    """Reschedule an existing appointment to a new date/time. Returns True if a row was updated."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            UPDATE appointments
            SET appointment_date = ?, appointment_time = ?
            WHERE id = ? AND status = 'CONFIRMED'
            """,
            (appointment_date, appointment_time, appointment_id),
        )
        return cursor.rowcount > 0


def cancel_appointment(appointment_id: int) -> bool:
    """Mark an appointment as CANCELLED rather than deleting it, to preserve history."""
    with get_connection() as conn:
        cursor = conn.execute(
            "UPDATE appointments SET status = 'CANCELLED' WHERE id = ?",
            (appointment_id,),
        )
        return cursor.rowcount > 0


def get_appointment(appointment_id: int):
    """Fetch a single appointment by id, or None if it doesn't exist."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM appointments WHERE id = ?", (appointment_id,)
        ).fetchone()
        return dict(row) if row else None


def list_appointments(business_type: str = None) -> list:
    """Return all appointments, most recent first, optionally filtered by business type."""
    with get_connection() as conn:
        if business_type:
            rows = conn.execute(
                "SELECT * FROM appointments WHERE business_type = ? ORDER BY created_at DESC",
                (business_type,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM appointments ORDER BY created_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]


def get_metrics(business_type: str = None) -> dict:
    """Aggregate counts used by the dashboard metrics section of the UI."""
    rows = list_appointments(business_type)
    total = len(rows)
    confirmed = sum(1 for r in rows if r["status"] == "CONFIRMED")
    cancelled = sum(1 for r in rows if r["status"] == "CANCELLED")
    return {"total": total, "confirmed": confirmed, "cancelled": cancelled}


# Initialise the database as soon as this module is imported, so the
# app "just works" on a fresh clone without a separate setup step.
init_db()
