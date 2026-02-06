# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


def nested_update(d: dict, u: dict) -> dict:
    """Update a nested dictionary with another nested dictionary."""
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            nested_update(d[k], v)
        else:
            d[k] = v
    return d
