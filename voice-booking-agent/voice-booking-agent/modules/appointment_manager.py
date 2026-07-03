"""
modules/appointment_manager.py
--------------------------------
Thin business-logic layer that sits between the conversation manager
/ Streamlit UI and the database module. Centralising it here means
the UI code never talks to SQLite directly, and the validation rules
used at booking time are also reused for rescheduling.
"""

from modules import database
from modules.validator import validate_date, validate_time
from modules.utils import get_logger

logger = get_logger(__name__)


class AppointmentManager:
    def __init__(self, config: dict):
        self.config = config

    # ------------------------------------------------------------------ #
    # Create
    # ------------------------------------------------------------------ #
    def book_appointment(self, answers: dict) -> int:
        """Persist a completed booking and return the new appointment id."""
        appointment_id = database.insert_appointment(answers)
        logger.info("Booked appointment #%s for %s", appointment_id, answers.get("full_name"))
        return appointment_id

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #
    def get_history(self, business_type: str = None) -> list:
        return database.list_appointments(business_type)

    def get_metrics(self, business_type: str = None) -> dict:
        return database.get_metrics(business_type)

    def get_appointment(self, appointment_id: int):
        return database.get_appointment(appointment_id)

    # ------------------------------------------------------------------ #
    # Update / Cancel (optional features)
    # ------------------------------------------------------------------ #
    def reschedule_appointment(self, appointment_id: int, new_date: str, new_time: str):
        """
        Validate the new date/time against the same rules used at
        booking time, then update the record. Returns (ok, message).
        """
        working_hours = self.config.get("working_hours")

        ok_date, clean_date = validate_date(new_date)
        if not ok_date:
            return False, "Invalid or past date. Please provide a valid future date."

        ok_time, clean_time = validate_time(new_time, working_hours=working_hours)
        if not ok_time:
            return False, "Invalid time, or outside working hours."

        existing = database.get_appointment(appointment_id)
        if not existing:
            return False, f"No appointment found with ID {appointment_id}."
        if existing["status"] != "CONFIRMED":
            return False, "Only confirmed appointments can be rescheduled."

        updated = database.update_appointment(appointment_id, clean_date, clean_time)
        if not updated:
            return False, "Could not reschedule this appointment."
        return True, {"appointment_date": clean_date, "appointment_time": clean_time}

    def cancel_appointment(self, appointment_id: int):
        existing = database.get_appointment(appointment_id)
        if not existing:
            return False, f"No appointment found with ID {appointment_id}."
        if existing["status"] == "CANCELLED":
            return False, "This appointment is already cancelled."

        cancelled = database.cancel_appointment(appointment_id)
        if not cancelled:
            return False, "Could not cancel this appointment."
        return True, "Appointment cancelled."
