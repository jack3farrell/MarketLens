"""Date validation and parsing utilities."""
from __future__ import annotations
from datetime import datetime, date


DATE_FORMAT = "%Y-%m-%d"


def parse_date(date_str: str) -> date:
    """Parse a YYYY-MM-DD string into a date object. Raises ValueError on bad input."""
    try:
        return datetime.strptime(date_str, DATE_FORMAT).date()
    except ValueError:
        raise ValueError(f"Invalid date format: '{date_str}'. Expected YYYY-MM-DD.")


def validate_date_range(start: str, end: str) -> None:
    """
    Validate that start and end are valid dates and start < end.
    Raises ValueError with a descriptive message on failure.
    """
    start_date = parse_date(start)
    end_date = parse_date(end)

    if start_date >= end_date:
        raise ValueError(
            f"start date ({start}) must be before end date ({end})."
        )

    today = date.today()
    if start_date > today:
        raise ValueError(f"start date ({start}) cannot be in the future.")