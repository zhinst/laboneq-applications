# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Analysis for the SNAP gate calibration experiment.

Part 1 - Amplitude analysis:
    Fits the transmon response vs. selective drive amplitude to a
    sin^2 model. Both SNAP pulses rotate in the same direction.
    Together they complete a 2π rotation at the π amplitude, returning
    the transmon to |g>:

        P(e) = contrast * sin^2(pi * amp / amp_pi) + offset

Part 2 - Phase analysis:
    The experiment is now a 2-D sweep: outer over the SNAP drive
    phase phi, inner over the Wigner displacement phase arg(beta) at
    fixed |beta| = sqrt(n). The data shape per qubit is
    (n_phi, n_beta_phase).

    Stage 1 -- for each phi, fit the Wigner fringe vs arg(beta) to a
    single-period cosine and extract its angular position theta(phi):

        W(arg_beta) = A * cos(arg_beta - theta) + B

    Stage 2 -- fit the extracted fringe angles vs phi to a line:

        theta(phi) = slope * phi + intercept

    The calibration constant (drive-phase -> cavity-phase mapping) is
    the ``intercept``. ``slope`` is a validity check and should come
    out close to +/- 1; a slope far from +/-1 indicates a readout or
    state-prep problem rather than a good calibration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from laboneq import workflow

from laboneq_applications.analysis.fitting_helpers import fit_data_lmfit
from laboneq_applications.analysis.options import BasePlottingOptions
from laboneq_applications.conversion import (
    as_quantum_element_list,
    sweeps_per_quantum_element,
)

if TYPE_CHECKING:
    import matplotlib as mpl
    from laboneq.workflow.tasks.run_experiment import (
        RunExperimentResults,
    )

    from laboneq_applications.typing import (
        QuantumElements,
        QubitSweepPoints,
    )


# Default Part 2 sweep grids. The experiment module uses these too, so the
# analysis reshapes the data with the same axes that were swept.
DEFAULT_PHASES_TRANSMON = np.linspace(0, 2 * np.pi, 13, endpoint=False)
DEFAULT_PHASES_WIGNER = np.linspace(0, 2 * np.pi, 48, endpoint=False)


# ═════════════════════════════════════════════════════════════
# Options
# ═════════════════════════════════════════════════════════════


@workflow.workflow_options
class SnapAnalysisWorkflowOptions:
    """Options for the SNAP amplitude and phase analysis workflows.

    Attributes:
        do_plotting:
            Whether to create plots.
            Default: True.
    """

    do_plotting: bool = workflow.option_field(
        True, description="Whether to create plots."
    )


@workflow.task_options
class SnapAmplitudeFitDataOptions:
    """Options for the `fit_amplitude_data` task (Part 1).

    Attributes:
        num_fit_points:
            Number of points for the fit curve. Default: 200.
    """

    num_fit_points: int = workflow.option_field(
        200, description="Number of points for the fit curve."
    )


@workflow.task_options
class SnapPhaseFitDataOptions:
    """Options for the `fit_phase_data` task (Part 2).

    Attributes:
        num_fit_points:
            Number of points for the fit curve. Default: 200.
        min_fringe_r2:
            Minimum R^2 for a Stage-1 fringe fit to be trusted and
            included in the Stage-2 line fit. Default: 0.5.
        max_slope_dev:
            Maximum allowed deviation of |slope| from 1.0 in the
            Stage-2 fit before the calibration is flagged as
            unreliable. Default: 0.3.
    """

    num_fit_points: int = workflow.option_field(
        200, description="Number of points for the fit curve."
    )
    min_fringe_r2: float = workflow.option_field(
        0.5, description="Minimum R^2 for a fringe fit to enter the line fit."
    )
    max_slope_dev: float = workflow.option_field(
        0.3, description="Maximum allowed deviation of |slope| from 1."
    )


# ═════════════════════════════════════════════════════════════
# Workflows
# ═════════════════════════════════════════════════════════════


@workflow.workflow
def amplitude_analysis_workflow(
    result: RunExperimentResults,
    qubits: QuantumElements,
    amplitudes_transmon: QubitSweepPoints,
    photon_number_memory: int,
    options: SnapAnalysisWorkflowOptions | None = None,
) -> None:
    """The SNAP amplitude analysis workflow (Part 1).

    The workflow consists of the following steps:

    - [fit_amplitude_data][]
    - [extract_amplitude_parameters][]
    - [plot_amplitude_calibration][]

    Arguments:
        result:
            The experiment results returned by the run_amplitude_experiment task.
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the result.
        amplitudes_transmon:
            The selective drive amplitudes that were swept for each qubit.
        photon_number_memory:
            The photon number that was calibrated.
        options:
            The options for building the workflow, passed as an instance of
            [SnapAnalysisWorkflowOptions][].

    Returns:
        WorkflowBuilder:
            The builder for the analysis workflow.
    """
    processed_data_dict = fit_amplitude_data(
        qubits, result, amplitudes_transmon, photon_number_memory
    )
    qubit_parameters = extract_amplitude_parameters(qubits, processed_data_dict)
    with workflow.if_(options.do_plotting):
        plot_amplitude_calibration(qubits, processed_data_dict)
    workflow.return_(qubit_parameters)


@workflow.workflow
def phase_analysis_workflow(
    result: RunExperimentResults,
    qubits: QuantumElements,
    photon_number_memory: int,
    phases_transmon: QubitSweepPoints | None = None,
    phases_wigner: QubitSweepPoints | None = None,
    options: SnapAnalysisWorkflowOptions | None = None,
) -> None:
    """The SNAP phase analysis workflow (Part 2).

    The workflow consists of the following steps:

    - [fit_phase_data][]
    - [extract_phase_parameters][]
    - [plot_phase_calibration][]

    Arguments:
        result:
            The experiment results returned by the run_phase_experiment task.
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the result.
        photon_number_memory:
            The photon number that was calibrated.
        phases_transmon:
            The SNAP drive phases phi that were swept (outer axis). None uses
            the experiment's default grid.
        phases_wigner:
            The Wigner displacement phases arg(beta) that were swept (inner
            axis). None uses the experiment's default grid.
        options:
            The options for building the workflow, passed as an instance of
            [SnapAnalysisWorkflowOptions][].

    Returns:
        WorkflowBuilder:
            The builder for the analysis workflow.
    """
    processed_data_dict = fit_phase_data(
        qubits, result, photon_number_memory, phases_transmon, phases_wigner
    )
    qubit_parameters = extract_phase_parameters(qubits, processed_data_dict)
    with workflow.if_(options.do_plotting):
        plot_phase_calibration(qubits, processed_data_dict)
    workflow.return_(qubit_parameters)


# ═════════════════════════════════════════════════════════════
# Analysis tasks
# ═════════════════════════════════════════════════════════════


@workflow.task
def fit_amplitude_data(
    qubits: QuantumElements,
    result: RunExperimentResults,
    amplitudes_transmon: QubitSweepPoints,
    photon_number_memory: int,
    options: SnapAmplitudeFitDataOptions | None = None,
) -> dict[str, dict]:
    """Fit Part 1: extract the pi amplitude for SNAP.

    Fits a sin^2 model to the two-pulse Rabi data and extracts the pi
    amplitude for the given (qubit, photon number).

    Arguments:
        qubits:
            The qubits on which to run the analysis.
        result:
            The experiment results returned by the run_amplitude_experiment task.
        amplitudes_transmon:
            The selective drive amplitudes that were swept for each qubit.
        photon_number_memory:
            The photon number that was calibrated.
        options:
            The options for this task as an instance of
            [SnapAmplitudeFitDataOptions][].

    Returns:
        Dictionary keyed by qubit UID with the structure:

            ```python
            {
                "q0": {
                    "amp_pi_snap": {"3": 0.45},
                    "rabi_data": {...},
                    "amplitudes": array,
                    "photon_number": 3,
                },
            }
            ```
    """
    opts = SnapAmplitudeFitDataOptions() if options is None else options
    qubits = as_quantum_element_list(qubits)
    n = photon_number_memory

    processed_data_dict = {}
    for q, q_amps_raw in sweeps_per_quantum_element(qubits, amplitudes_transmon):
        q_amps = np.asarray(q_amps_raw)

        handle = _snap_amp_handle(q.uid, n)
        data_raw = result.get_data(handle)
        data = np.real(np.asarray(data_raw))

        fit_result = _fit_snap_rabi(q_amps, data, opts.num_fit_points)

        rabi_data = {
            "raw_data": data,
            "fit_params": list(fit_result["params"]),
            "fit_success": bool(fit_result["success"]),
            "amp_pi": float(fit_result["amp_pi"]),
            "fit_x": fit_result["fit_x"],
            "fit_y": fit_result["fit_y"],
        }

        processed_data_dict[q.uid] = {
            "amp_pi_snap": {str(n): float(fit_result["amp_pi"])},
            "rabi_data": rabi_data,
            "amplitudes": q_amps,
            "photon_number": n,
        }

    return processed_data_dict


@workflow.task
def extract_amplitude_parameters(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict],
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Extract the SNAP pi amplitude from the Part 1 fit results.

    Arguments:
        qubits:
            The qubits on which to run the analysis.
        processed_data_dict:
            The processed data dictionary returned by fit_amplitude_data.

    Returns:
        dict with the extracted qubit parameters and their previous values:

            ```python
            {
                "new_parameter_values": {
                    q.uid: {"snap_drive_amp_per_n": {..., str(n): amp_pi}},
                },
                "old_parameter_values": {
                    q.uid: {"snap_drive_amp_per_n": {...}},
                },
            }
            ```

            The new dictionary is a copy of the old one with the entry for the
            calibrated photon number set. If the fit failed for a qubit, its
            new_parameter_values entry is left empty.
    """
    qubits = as_quantum_element_list(qubits)
    qubit_parameters = {
        "old_parameter_values": {q.uid: {} for q in qubits},
        "new_parameter_values": {q.uid: {} for q in qubits},
    }
    for q in qubits:
        old = dict(q.parameters.snap_drive_amp_per_n)
        qubit_parameters["old_parameter_values"][q.uid] = {"snap_drive_amp_per_n": old}
        res = processed_data_dict[q.uid]
        n_str = str(res["photon_number"])
        amp_pi = res["amp_pi_snap"][n_str]
        if res["rabi_data"]["fit_success"] and np.isfinite(amp_pi):
            qubit_parameters["new_parameter_values"][q.uid] = {
                "snap_drive_amp_per_n": {**old, n_str: amp_pi},
            }
    return qubit_parameters


@workflow.task
def fit_phase_data(
    qubits: QuantumElements,
    result: RunExperimentResults,
    photon_number_memory: int,
    phases_transmon: QubitSweepPoints | None = None,
    phases_wigner: QubitSweepPoints | None = None,
    options: SnapPhaseFitDataOptions | None = None,
) -> dict[str, dict]:
    """Fit Part 2: extract the SNAP phase calibration.

    For each qubit the raw data is reshaped to (n_phi, n_beta_phase).
    Stage 1 fits each fringe (a row) for its angle theta(phi); Stage 2
    fits theta(phi) vs phi to a line. The intercept is the calibration
    constant.

    Arguments:
        qubits:
            The qubits on which to run the analysis.
        result:
            The experiment results returned by the run_phase_experiment task.
        photon_number_memory:
            The photon number that was calibrated.
        phases_transmon:
            The SNAP drive phases phi that were swept (outer axis). None uses
            the experiment's default grid.
        phases_wigner:
            The Wigner displacement phases arg(beta) that were swept (inner
            axis). None uses the experiment's default grid.
        options:
            The options for this task as an instance of [SnapPhaseFitDataOptions][].

    Returns:
        Dictionary keyed by qubit UID with the structure:

            ```python
            {
                "q0": {
                    "phase_offsets": {"3": <intercept, rad>},
                    "phase_slope_signs": {"3": <sign of slope, +/-1>},
                    "calibration": {
                        "slope": ...,
                        "intercept": ...,
                        "slope_ok": bool,
                        "n_good_fringes": int,
                    },
                    "fringe_angles": array,   # theta(phi)
                    "fringe_r2": array,       # R^2 of each fringe fit
                    "phases": array,          # phi
                    "phases_wigner": array,   # arg(beta)
                    "data_2d": array,         # (n_phi, n_beta_phase)
                    "line_fit": dict,         # Stage-2 fit details
                    "photon_number": 3,
                },
            }
            ```
    """
    opts = SnapPhaseFitDataOptions() if options is None else options
    qubits = as_quantum_element_list(qubits)
    n = photon_number_memory

    phases_per_qubit = broadcast_phases(
        phases_transmon,
        len(qubits),
        default=DEFAULT_PHASES_TRANSMON,
    )
    phases_wigner_per_qubit = broadcast_phases(
        phases_wigner,
        len(qubits),
        default=DEFAULT_PHASES_WIGNER,
    )

    processed_data_dict = {}

    for q, phis_raw, betas_raw in zip(
        qubits,
        phases_per_qubit,
        phases_wigner_per_qubit,
        strict=False,
    ):
        phis = np.asarray(phis_raw).ravel()
        betas = np.asarray(betas_raw).ravel()
        n_phi = phis.size
        n_beta = betas.size

        handle = _snap_phase_handle(q.uid, n)
        data_raw = result.get_data(handle)
        data = np.real(np.asarray(data_raw))

        # Reshape to (n_phi, n_beta). LabOne returns the 2-D sweep
        # with the outer axis (phi) first; reshape also handles a
        # flattened return.
        data = data.reshape(n_phi, n_beta)

        # ---- Stage 1: fringe angle per phi ----
        fringe_angles = np.full(n_phi, np.nan)
        fringe_r2 = np.full(n_phi, np.nan)
        for i in range(n_phi):
            fr = _fit_fringe(betas, data[i], opts.num_fit_points)
            fringe_angles[i] = fr["angle"]
            fringe_r2[i] = fr["r2"]

        # ---- Stage 2: theta(phi) vs phi line fit ----
        good = fringe_r2 >= opts.min_fringe_r2
        line = _fit_phase_line(phis[good], fringe_angles[good])
        slope_ok = abs(abs(line["slope"]) - 1.0) <= opts.max_slope_dev

        processed_data_dict[q.uid] = {
            "phase_offsets": {str(n): float(line["intercept"])},
            "phase_slope_signs": {str(n): float(np.sign(line["slope"]))},
            "calibration": {
                "slope": float(line["slope"]),
                "intercept": float(line["intercept"]),
                "slope_ok": bool(slope_ok),
                "n_good_fringes": int(good.sum()),
            },
            "fringe_angles": fringe_angles,
            "fringe_r2": fringe_r2,
            "phases": phis,
            "phases_wigner": betas,
            "data_2d": data,
            "line_fit": line,
            "photon_number": n,
        }

    return processed_data_dict


@workflow.task
def extract_phase_parameters(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict],
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Extract the SNAP phase offset and slope sign from the Part 2 fit results.

    Arguments:
        qubits:
            The qubits on which to run the analysis.
        processed_data_dict:
            The processed data dictionary returned by fit_phase_data.

    Returns:
        dict with the extracted qubit parameters and their previous values:

            ```python
            {
                "new_parameter_values": {
                    q.uid: {
                        "snap_drive_phase_per_n": {..., str(n): intercept},
                        "snap_drive_phase_slope_sign_per_n": {..., str(n): sign},
                    },
                },
                "old_parameter_values": {...},  # same form
            }
            ```

            The new dictionaries are copies of the old ones with the entries for
            the calibrated photon number set. If the slope check failed for a
            qubit, its new_parameter_values entry is left empty.
    """
    qubits = as_quantum_element_list(qubits)
    qubit_parameters = {
        "old_parameter_values": {q.uid: {} for q in qubits},
        "new_parameter_values": {q.uid: {} for q in qubits},
    }
    for q in qubits:
        old_phase = dict(q.parameters.snap_drive_phase_per_n)
        old_sign = dict(q.parameters.snap_drive_phase_slope_sign_per_n)
        qubit_parameters["old_parameter_values"][q.uid] = {
            "snap_drive_phase_per_n": old_phase,
            "snap_drive_phase_slope_sign_per_n": old_sign,
        }
        res = processed_data_dict[q.uid]
        n_str = str(res["photon_number"])
        if res["calibration"]["slope_ok"]:
            qubit_parameters["new_parameter_values"][q.uid] = {
                "snap_drive_phase_per_n": {
                    **old_phase,
                    n_str: res["phase_offsets"][n_str],
                },
                "snap_drive_phase_slope_sign_per_n": {
                    **old_sign,
                    n_str: res["phase_slope_signs"][n_str],
                },
            }
    return qubit_parameters


@workflow.task
def plot_amplitude_calibration(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict],
    options: BasePlottingOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Plot Part 1: the two-pulse Rabi data and fit.

    Arguments:
        qubits:
            The qubits on which to run the analysis.
        processed_data_dict:
            The processed data dictionary returned by fit_amplitude_data.
        options:
            The options for this task as an instance of [BasePlottingOptions][].

    Returns:
        dict with qubit UIDs as keys and the figures for each qubit as values.
    """
    opts = BasePlottingOptions() if options is None else options
    qubits = as_quantum_element_list(qubits)
    figures = {}
    for q in qubits:
        res = processed_data_dict[q.uid]
        n = res["photon_number"]
        fig = _plot_amplitude_calibration(q.uid, res["amplitudes"], res["rabi_data"], n)
        _save_and_close(fig, f"snap_amp_cal_{q.uid}_n{n}", opts)
        figures[q.uid] = fig
    return figures


@workflow.task
def plot_phase_calibration(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict],
    options: BasePlottingOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Plot Part 2: the 2-D Wigner fringe map and the theta(phi) line fit.

    Arguments:
        qubits:
            The qubits on which to run the analysis.
        processed_data_dict:
            The processed data dictionary returned by fit_phase_data.
        options:
            The options for this task as an instance of [BasePlottingOptions][].

    Returns:
        dict with qubit UIDs as keys and the figures for each qubit as values.
    """
    opts = BasePlottingOptions() if options is None else options
    qubits = as_quantum_element_list(qubits)
    figures = {}
    for q in qubits:
        res = processed_data_dict[q.uid]
        n = res["photon_number"]
        fig = _plot_phase_calibration(
            q.uid,
            res["phases"],
            res["phases_wigner"],
            res["data_2d"],
            res["fringe_angles"],
            res["line_fit"],
            n,
        )
        _save_and_close(fig, f"snap_phase_cal_{q.uid}_n{n}", opts)
        figures[q.uid] = fig
    return figures


def _save_and_close(
    fig: mpl.figure.Figure,
    name: str,
    opts: BasePlottingOptions,
) -> None:
    if opts.save_figures:
        workflow.save_artifact(name, fig)
    if opts.close_figures:
        plt.close(fig)


# ═════════════════════════════════════════════════════════════
# Fitting helpers - Part 1
# ═════════════════════════════════════════════════════════════


def _snap_rabi_model(
    x: np.ndarray,
    amp_pi: float,
    contrast: float,
    offset: float,
) -> np.ndarray:
    """Two-pulse SNAP Rabi model: contrast*sin^2(pi*x/amp_pi)+offset."""
    return contrast * np.sin(np.pi * x / amp_pi) ** 2 + offset


def _guess_snap_rabi_params(
    amplitudes: np.ndarray,
    data: np.ndarray,
    num_candidates: int = 400,
) -> tuple[float, float, float]:
    """Initial guess (amp_pi, contrast, offset) for the SNAP Rabi fit.

    The sin^2 model is ambiguous for a local optimiser started from a
    heuristic guess, so amp_pi is grid-searched instead: for each
    candidate, contrast and offset follow from linear least squares, and
    the candidate with the lowest residual is kept. The contrast may be
    negative, since the sign of the integrated signal is arbitrary.

    A dip at amp_pi with positive contrast looks like a dip at 2*amp_pi
    with negative contrast, so amp_pi is restricted to the swept range,
    which the sweep is meant to be centred on.
    """
    best = (np.inf, float(amplitudes[-1]), 0.0, float(np.mean(data)))
    for amp_pi in np.linspace(*_amp_pi_bounds(amplitudes), num_candidates):
        design = np.column_stack(
            [np.sin(np.pi * amplitudes / amp_pi) ** 2, np.ones_like(amplitudes)]
        )
        (contrast, offset), *_ = np.linalg.lstsq(design, data, rcond=None)
        residual = float(np.sum((design @ (contrast, offset) - data) ** 2))
        if residual < best[0]:
            best = (residual, float(amp_pi), float(contrast), float(offset))
    return best[1:]


def _amp_pi_bounds(amplitudes: np.ndarray) -> tuple[float, float]:
    """Allowed (min, max) of amp_pi: the swept amplitude range.

    Raises:
        ValueError: If the sweep has no positive range to fit.
    """
    amp_pi_min = max(float(np.min(amplitudes)), 1e-6)
    amp_pi_max = float(np.max(amplitudes))
    if amp_pi_max <= amp_pi_min:
        raise ValueError("The amplitude sweep has no positive range to fit.")
    return amp_pi_min, amp_pi_max


def _fit_snap_rabi(
    amplitudes: np.ndarray,
    data: np.ndarray,
    num_fit_points: int = 200,
) -> dict:
    """Fit two-pulse SNAP Rabi and extract pi amplitude."""
    fit_x = np.linspace(amplitudes[0], amplitudes[-1], num_fit_points)
    try:
        amp_pi_min, amp_pi_max = _amp_pi_bounds(amplitudes)
        amp_pi_guess, contrast_guess, offset_guess = _guess_snap_rabi_params(
            amplitudes, data
        )

        param_hints = {
            "amp_pi": {"value": amp_pi_guess, "min": amp_pi_min, "max": amp_pi_max},
            "contrast": {"value": contrast_guess},
            "offset": {"value": offset_guess},
        }
        fit_res = fit_data_lmfit(
            _snap_rabi_model,
            amplitudes,
            data,
            param_hints=param_hints,
        )
        popt = np.array(
            [fit_res.best_values[k] for k in ("amp_pi", "contrast", "offset")]
        )
        success = bool(fit_res.success)
        fit_y = _snap_rabi_model(fit_x, *popt)
    except (RuntimeError, ValueError, TypeError):
        popt = np.array([np.nan, 0.0, float(np.mean(data))])
        success = False
        fit_y = np.full_like(fit_x, np.nan)

    amp_pi = float(popt[0])

    return {
        "params": tuple(popt),
        "success": success,
        "amp_pi": amp_pi,
        "fit_x": fit_x,
        "fit_y": fit_y,
    }


# ═════════════════════════════════════════════════════════════
# Fitting helpers - Part 2
# ═════════════════════════════════════════════════════════════


def _fringe_model(
    x: np.ndarray,
    amplitude: float,
    angle: float,
    offset: float,
) -> np.ndarray:
    """Single-period Wigner fringe vs arg(beta).

    W(arg_beta) = amplitude * cos(arg_beta - angle) + offset

    with ``x`` the Wigner displacement phase arg(beta). ``angle`` is the
    angular position of the fringe maximum -- the cavity-phase observable
    for a given SNAP drive phase.
    """
    return amplitude * np.cos(x - angle) + offset


def _r_squared(y: np.ndarray, y_fit: np.ndarray) -> float:
    """Coefficient of determination."""
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _fit_fringe(
    beta_phases: np.ndarray,
    data: np.ndarray,
    num_fit_points: int = 200,
) -> dict:
    """Fit one Wigner fringe vs arg(beta); return its angle.

    The angle guess uses the first Fourier harmonic, which is robust
    against noise compared with a single argmax.
    """
    try:
        offset_guess = float(np.mean(data))
        amp_guess = float((np.max(data) - np.min(data)) / 2)

        # First-harmonic phase estimate for the angle guess.
        c = np.sum(data * np.cos(beta_phases))
        s = np.sum(data * np.sin(beta_phases))
        angle_guess = float(np.arctan2(s, c))

        param_hints = {
            "amplitude": {"value": amp_guess},
            "angle": {"value": angle_guess},
            "offset": {"value": offset_guess},
        }
        fit_res = fit_data_lmfit(
            _fringe_model,
            beta_phases,
            data,
            param_hints=param_hints,
        )
        popt = np.array(
            [fit_res.best_values[k] for k in ("amplitude", "angle", "offset")]
        )
        # Enforce non-negative amplitude; fold sign into the angle.
        if popt[0] < 0:
            popt[0] = -popt[0]
            popt[1] = popt[1] + np.pi
        success = bool(fit_res.success)
    except (RuntimeError, ValueError, TypeError):
        popt = np.array([0.0, 0.0, float(np.mean(data))])
        success = False

    angle = float(popt[1] % (2 * np.pi))
    fit_x = np.linspace(beta_phases[0], beta_phases[-1], num_fit_points)
    fit_y = _fringe_model(fit_x, *popt)
    r2 = _r_squared(data, _fringe_model(beta_phases, *popt))

    return {
        "params": tuple(popt),
        "success": success,
        "angle": angle,
        "r2": r2,
        "fit_x": fit_x,
        "fit_y": fit_y,
    }


def _fit_phase_line(phi: np.ndarray, theta: np.ndarray) -> dict:
    """Fit theta(phi) vs phi to a line, handling 2*pi wrapping.

    Unwraps theta after sorting by phi so the slope (expected ~ +/-1)
    is not corrupted by branch jumps.
    """
    if phi.size < 2:  # noqa: PLR2004
        return {"slope": float("nan"), "intercept": float("nan")}

    order = np.argsort(phi)
    phi_s = phi[order]
    theta_s = np.unwrap(theta[order])

    slope, intercept = np.polyfit(phi_s, theta_s, 1)
    return {
        "slope": float(slope),
        "intercept": float(intercept % (2 * np.pi)),
        "intercept_raw": float(intercept),
        "phi_sorted": phi_s,
        "theta_unwrapped": theta_s,
    }


# ═════════════════════════════════════════════════════════════
# Plotting helpers
# ═════════════════════════════════════════════════════════════


def _plot_amplitude_calibration(
    q_uid: str,
    amplitudes: np.ndarray,
    rabi_data: dict,
    photon_number: int,
) -> plt.Figure:
    """Plot Part 1: two-pulse Rabi for a single photon number."""
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(
        amplitudes, rabi_data["raw_data"], "o", markersize=3, alpha=0.6, label="Data"
    )
    ax.plot(rabi_data["fit_x"], rabi_data["fit_y"], "-", linewidth=1.5, label="Fit")
    ax.axvline(
        rabi_data["amp_pi"],
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"amp_pi = {rabi_data['amp_pi']:.4f}",
    )
    ax.set_xlabel("Selective drive amplitude")
    ax.set_ylabel("Signal")
    ax.set_title(
        f"{q_uid} - SNAP amplitude calibration (Part 1) - n = {photon_number}",
        fontweight="bold",
    )
    ax.legend(fontsize=8)
    return fig


def _plot_phase_calibration(
    q_uid: str,
    phis: np.ndarray,
    betas: np.ndarray,
    data_2d: np.ndarray,
    fringe_angles: np.ndarray,
    line: dict,
    photon_number: int,
) -> plt.Figure:
    """Plot Part 2: 2-D Wigner map + theta(phi) line fit."""
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    # Left: 2-D Wigner fringe map with extracted angles overlaid.
    im = ax0.pcolormesh(
        np.degrees(betas),
        np.degrees(phis),
        data_2d,
        shading="auto",
    )
    ax0.plot(
        np.degrees(fringe_angles),
        np.degrees(phis),
        "w.",
        markersize=5,
        label="fringe angle theta(phi)",
    )
    ax0.set_xlabel("Wigner displacement phase arg(beta) (deg)")
    ax0.set_ylabel("SNAP drive phase phi (deg)")
    ax0.set_title(f"{q_uid} - Wigner fringes - n = {photon_number}", fontweight="bold")
    ax0.legend(fontsize=8, loc="upper right")
    fig.colorbar(im, ax=ax0, label="Wigner signal")

    # Right: theta(phi) vs phi with the line fit.
    phi_s = line.get("phi_sorted")
    theta_s = line.get("theta_unwrapped")
    if phi_s is not None:
        ax1.plot(
            np.degrees(phi_s),
            np.degrees(theta_s),
            "o",
            markersize=4,
            label="theta(phi) (unwrapped)",
        )
        # Use the unfolded intercept so the line overlays the unwrapped
        # data rather than a 2*pi-shifted copy of it.
        fit_line = line["slope"] * phi_s + line["intercept_raw"]
        ax1.plot(
            np.degrees(phi_s),
            np.degrees(fit_line),
            "-",
            linewidth=1.5,
            label=(
                f"slope={line['slope']:.3f}, "
                f"intercept={np.degrees(line['intercept']):.1f} deg"
            ),
        )
        ax1.legend(fontsize=8)
    ax1.set_xlabel("SNAP drive phase phi (deg)")
    ax1.set_ylabel("Fringe angle theta (deg)")
    ax1.set_title(f"{q_uid} - phase calibration (Part 2)", fontweight="bold")

    return fig


# ═════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════


def broadcast_phases(
    phases: np.ndarray | list | None,
    n_qubits: int,
    default: np.ndarray,
) -> list[np.ndarray]:
    """Normalise a phase spec into a per-qubit list of 1-D arrays."""
    if phases is None:
        return [np.asarray(default) for _ in range(n_qubits)]

    is_list_of_arrays = (
        isinstance(phases, list)
        and len(phases) > 0
        and isinstance(phases[0], (list, tuple, np.ndarray))
    )
    if is_list_of_arrays:
        out = [np.asarray(p).ravel() for p in phases]
        if len(out) == 1 and n_qubits > 1:
            out = out * n_qubits
        if len(out) != n_qubits:
            raise ValueError(
                "phase spec must have one array per qubit "
                f"(got {len(out)}, expected {n_qubits})."
            )
        return out

    single = np.asarray(phases).ravel()
    return [single for _ in range(n_qubits)]


def _snap_amp_handle(qubit_uid: str, n: int) -> str:
    """Result handle for SNAP amplitude measurement."""
    return f"{qubit_uid}/snap_amp_n{n}"


def _snap_phase_handle(qubit_uid: str, n: int) -> str:
    """Result handle for SNAP phase measurement."""
    return f"{qubit_uid}/snap_phase_n{n}"
