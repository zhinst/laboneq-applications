"""Utility functions for the core module."""

from __future__ import annotations

from datetime import datetime, timezone


def now(timestamp: datetime | None = None) -> datetime:
    """Returns the given timestamp or the current time.

    Uses UTC timezone.
    """
    return (timestamp or datetime.now(timezone.utc)).astimezone(timezone.utc)
