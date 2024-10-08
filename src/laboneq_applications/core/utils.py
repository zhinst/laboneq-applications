"""Utility functions for the core module."""

from __future__ import annotations

import functools
from datetime import datetime, timezone
from typing import Callable

import pygments
import pygments.formatters
import pygments.lexers
from dateutil.tz import tzlocal


def utc_now(timestamp: datetime | None = None) -> datetime:
    """Returns the given timestamp or the current time.

    Uses UTC timezone.
    """
    return (timestamp or datetime.now(timezone.utc)).astimezone(timezone.utc)


def local_now(dt: datetime | None = None) -> datetime:
    """Returns the given datetime in the local timezone.

    Arguments:
        dt:
            The time to convert to the local timezone. If None,
            the current time is used.

    Returns:
        The time in the local timezone.
    """
    return (dt or datetime.now(tzlocal())).astimezone(tzlocal())


def local_timestamp(dt: datetime | None = None) -> str:
    """Returns a string formatted timestamp in the local timezone.

    Arguments:
        dt:
            The time use. If None, the current time is used.

    Returns:
        The datetime formatted as '%Y%m%dT%H%M%S'.

    Note:
        This function is used by the `FolderStore` to generate the
        timestamp used in the names of the logbook folders it creates.
    """
    return local_now(dt).strftime("%Y%m%dT%H%M%S")


def local_date_stamp(dt: datetime | None = None) -> str:
    """Returns a string formatted date stamp in the local timezone.

    Arguments:
        dt:
            The time use. If None, the current time is used.

    Returns:
        THe datetime formatted as '%Y%m%d'.

    Note:
        This function is used by the `FolderStore` to generate the
        date stamp used in the names of the per-day subfolders it
        creates.
    """
    return local_now(dt).strftime("%Y%m%d")


class PygmentedStr(str):
    """A sub-class of `str` that renders highlighted code in IPython notebooks."""

    __slots__ = ("lexer",)

    def __new__(cls, src: str, *, lexer: pygments.lexer.Lexer):
        """Build a new instance."""
        instance = super().__new__(cls, src)
        instance.lexer = lexer
        return instance

    def _repr_html_(self) -> str:
        return pygments.highlight(self, self.lexer, pygments.formatters.HtmlFormatter())


def pygmentize(
    f: Callable[..., str] | None = None, lexer: pygments.lexer.Lexer | str = "python"
) -> Callable[..., str]:
    """A decorator for adding pygments syntax highlighting in Jupyter notebooks.

    Arguments:
        f:
            The function to decorate.
        lexer:
            The name of the pygments lexer to use.

    Return:
        If `f` is not None, the decorated function. Otherwise, a new decorator
        with partial arguments supplied.
    """
    if f is None:
        return functools.partial(pygmentize, lexer=lexer)
    if isinstance(lexer, str):
        lexer = pygments.lexers.get_lexer_by_name(lexer)

    @functools.wraps(f)
    def pygments_wrapper(*args, **kw) -> PygmentedStr:
        result = f(*args, **kw)
        return PygmentedStr(result, lexer=lexer)

    return pygments_wrapper
