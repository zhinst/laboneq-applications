from datetime import datetime, timezone

from laboneq_applications.logbook import format_time


def test_format_time():
    ts = datetime(2000, 1, 1, 15, tzinfo=timezone.utc)
    assert format_time(ts) == "2000-01-01 15:00:00.000000Z"
