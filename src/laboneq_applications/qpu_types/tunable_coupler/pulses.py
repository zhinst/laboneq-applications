# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Pulse shapes for tunable-coupler driven two-qubit gates."""

from __future__ import annotations

import numpy as np
from laboneq.simple import dsl


@dsl.pulse_library.register_pulse_functional
def modulated_flux(
    x: np.ndarray,
    frequency: float = 300e6,
    sigma: float = 1 / 3,
    width: float | None = None,
    *,
    length: float,
    zero_boundaries: bool = False,
    **_,
) -> np.ndarray:
    """Gaussian-square envelope modulated at a carrier frequency.

    The envelope has a flat top of duration ``width`` with Gaussian rise/fall on
    either side, multiplied by a sine at ``frequency``. Used to drive parametric
    two-qubit gates (e.g. ISWAP, CZ) via the tunable coupler's flux line.

    Arguments:
        x:
            Pulse sample positions in the normalised range [-1, 1], passed by
            the LabOne Q pulse-functional dispatch.
        length:
            Length of the pulse in seconds.
        frequency:
            Modulation frequency of the pulse in Hz.
        width:
            Width of the flat portion of the pulse in seconds. Defaults to 90%
            of ``length`` if not provided.
        sigma:
            Standard deviation of the Gaussian rise/fall portions (relative
            units; the Gaussian span is parametrised in [-1, 1]).
        zero_boundaries:
            If True, shifts and renormalises the envelope so it is exactly zero
            at the boundaries.

    Keyword Arguments:
        uid (str): Unique identifier of the pulse.
        amplitude (float): Amplitude of the pulse.

    Returns:
        The sampled pulse values.
    """
    if width is not None and width >= length:
        raise ValueError(
            "The width of the flat portion of the pulse must be smaller than "
            "the total length."
        )

    if width is None:
        width = 0.9 * length

    risefall_in_samples = round(len(x) * (1 - width / length) / 2)
    flat_in_samples = len(x) - 2 * risefall_in_samples
    gauss_x = np.linspace(-1.0, 1.0, 2 * risefall_in_samples)
    gauss_part = np.exp(-(gauss_x**2) / (2 * sigma**2))
    gauss_sq = np.concatenate(
        (
            gauss_part[:risefall_in_samples],
            np.ones(flat_in_samples),
            gauss_part[risefall_in_samples:],
        )
    )
    if zero_boundaries:
        t_left = gauss_x[0] - (gauss_x[1] - gauss_x[0])
        delta = np.exp(-(t_left**2) / (2 * sigma**2))
        gauss_sq -= delta
        gauss_sq /= 1 - delta

    gauss_sq *= np.sin(2 * np.pi * frequency * (0.5 * (x + 1) * length))

    return gauss_sq
