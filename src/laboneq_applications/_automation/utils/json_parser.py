# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import collections.abc

import numpy as np


def make_json_serializable(obj: object) -> object:
    """Recursively convert object to JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, np.ndarray):
        # Convert numpy arrays to lists
        return make_json_serializable(obj.tolist())
    if isinstance(obj, (np.integer, np.floating)):
        # Convert numpy scalar types to Python types
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    # Try to convert to list if it's array-like
    try:
        if isinstance(obj, collections.abc.Iterable) and not isinstance(
            obj, (str, bytes)
        ):
            return [make_json_serializable(item) for item in obj]
    except (TypeError, AttributeError):
        pass
    # Last resort: convert to string
    try:
        return str(obj)
    except TypeError:
        return f"<non-serializable: {type(obj).__name__}>"
