"""Utility functions for verifying pretty printing."""

import re


def minify_string(s):
    return s.replace("\n", "").replace(" ", "").replace("│", "").replace("↳", "")


def strip_ansi_codes(s):
    """Remove all ANSI codes that do formatting from the given string."""
    ansi_escape = re.compile(r"\x1b[[0-9;]*m")
    return ansi_escape.sub("", s)
