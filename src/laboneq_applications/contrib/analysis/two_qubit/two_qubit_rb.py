# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Analysis for the two-qubit randomized benchmarking experiment.

Data is acquired in SINGLE_SHOT + DISCRIMINATION mode, so each shot yields a
binary 0/1 result per qubit.  The per-shot bit outcomes of both qubits in a
pair are combined into a joint two-qubit state (``"00"``, ``"01"``, ``"10"``,
``"11"``), and the probability of each joint state is computed by averaging
over shots for each circuit in the sweep.

The ``"00"`` state probability is fitted with an exponential decay and the
error-per-Clifford (EPC) is extracted using the two-qubit depolarising
formula::

    EPC = (d - 1) / d * (1 - exp(-decay_rate))

where d = 2^2 = 4 for two qubits.

The analysis returns a dictionary of the form::

    {"q0_uid-q1_uid": {"epc_avg": float}}

which is the format expected by
``calibration_two_qubit_RB.evaluate_calibration``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import uncertainties as unc
from laboneq.analysis import fitting as fit_mods
from laboneq.dsl.quantum import QuantumElement
from laboneq.simple import dsl
from laboneq.workflow import (
    comment,
    if_,
    option_field,
    return_,
    save_artifact,
    task,
    task_options,
    workflow,
)
from uncertainties.umath import exp as uexp

from laboneq_applications.analysis.fitting_helpers import fit_data_lmfit
from laboneq_applications.analysis.options import (
    BasePlottingOptions,
    FitDataOptions,
    TuneUpAnalysisWorkflowOptions,
)
from laboneq_applications.core.validation import (
    validate_result,
)

if TYPE_CHECKING:
    import lmfit
    import matplotlib as mpl
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults
    from numpy.typing import ArrayLike

# Number of qubit states for the two-qubit Hilbert space: d = 2^2 = 4
_D_TWO_QUBIT = 4
_NDIM_2D = 2
_DISCRIMINATION_THRESHOLD = 0.5
# Joint state labels in bit-string order: q0 bit is MSB
_JOINT_STATES = ("00", "01", "10", "11")


@workflow
def analysis_workflow(
    result: RunExperimentResults,
    qubit_pairs: list[list[QuantumElement]],
    length_cliffords: list[int],
    variations: int,
    options: TuneUpAnalysisWorkflowOptions | None = None,
) -> None:
    """Two-qubit RB analysis workflow.

    The workflow consists of the following steps:

    - [calculate_qubit_population_tq_rb]()
    Combine per-shot qubit bits into joint two-qubit state populations.
    - [fit_data]()
    Fit an exponential decay to the P(00) population per pair.
    - [compute_pair_epc]()
    Compute pair EPC from the fit result.
    - [plot_population]()
    Optionally plot survival curves (one figure per pair).

    Arguments:
        result:
            The experiment results returned by the run_experiment task.
        qubit_pairs:
            List of [q0, q1] pairs that were benchmarked.
        length_cliffords:
            List of numbers of 2-qubit Clifford gates swept.
        variations:
            Number of random circuit samples per sequence length.
        options:
            Analysis workflow options.

    Returns:
        WorkflowBuilder:
            The builder for the analysis workflow.
    """
    processed_data_dict = calculate_qubit_population_tq_rb(
        qubit_pairs, result, length_cliffords, variations
    )
    fit_results = fit_data(qubit_pairs, processed_data_dict)
    pair_epc = compute_pair_epc(qubit_pairs, fit_results)

    with if_(options.do_plotting):
        with if_(options.do_qubit_population_plotting):
            plot_population(qubit_pairs, processed_data_dict, fit_results)

    return_(pair_epc)


def _normalize_raw(raw: np.ndarray, n_circuits: int) -> np.ndarray:
    """Return raw result data as a real-valued (n_circuits, n_shots) array.

    Handles the three layouts that LabOne Q may return:
    - 1D flat array of length ``n_circuits * n_shots``
    - 2D array ``(n_circuits, n_shots)``
    - 2D array ``(n_shots, n_circuits)`` (hardware SINGLE_SHOT layout)
    """
    raw = np.real(np.asarray(raw, dtype=complex))
    if raw.ndim == 1:
        n_shots = max(raw.size // n_circuits, 1)
        raw = raw.reshape(n_circuits, n_shots)
    elif (
        raw.ndim == _NDIM_2D
        and raw.shape[0] != n_circuits
        and raw.shape[1] == n_circuits
    ):
        raw = raw.T  # (n_shots, n_circuits) → (n_circuits, n_shots)
    return raw


@task
def calculate_qubit_population_tq_rb(
    qubit_pairs: list,
    result: RunExperimentResults,
    length_cliffords: list,
    variations: int,
) -> dict[str, dict[str, ArrayLike]]:
    """Compute joint two-qubit state populations from single-shot data.

    For each shot, the per-qubit discrimination bits (0 or 1) are combined
    into a joint two-qubit state label (``"00"``, ``"01"``, ``"10"``,
    ``"11"``).  The probability of each joint state is then obtained by
    averaging over shots for every circuit in the sweep.

    Arguments:
        qubit_pairs:
            List of [q0, q1] pairs.
        result:
            The experiment results (discrimination 0/1 per shot per qubit).
        length_cliffords:
            List of Clifford sequence lengths.
        variations:
            Number of random circuit samples per sequence length.

    Returns:
        Dictionary keyed by pair UID ``"q0_uid-q1_uid"``::

            {
                pair_key: {
                    "sweep_points": array,   # Clifford count per circuit
                    "00": array,             # P(q0=0, q1=0) per circuit
                    "01": array,             # P(q0=0, q1=1) per circuit
                    "10": array,             # P(q0=1, q1=0) per circuit
                    "11": array,             # P(q0=1, q1=1) per circuit
                }
            }
    """
    validate_result(result)

    # Sweep-point axis: [l0, l1, ..., l0, l1, ...] repeated `variations` times
    sweep_points = np.concatenate([length_cliffords for _ in range(variations)])
    n_circuits = len(sweep_points)

    processed_data_dict = {}
    for pair in qubit_pairs:
        q0, q1 = pair[0], pair[1]
        pair_key = f"{q0.uid}-{q1.uid}"

        raw0 = _normalize_raw(
            result[dsl.handles.result_handle(q0.uid)].data, n_circuits
        )
        raw1 = _normalize_raw(
            result[dsl.handles.result_handle(q1.uid)].data, n_circuits
        )

        # Binarise: threshold at 0.5 to handle both int (0/1) and float data
        bits0 = (raw0 > _DISCRIMINATION_THRESHOLD).astype(int)  # (n_circuits, n_shots)
        bits1 = (raw1 > _DISCRIMINATION_THRESHOLD).astype(int)

        n_shots = bits0.shape[1]
        # Encode joint state as integer: q0 is MSB → 0="00", 1="01", 2="10", 3="11"
        joint = bits0 * 2 + bits1  # (n_circuits, n_shots)

        populations = {
            state: (joint == idx).sum(axis=1) / n_shots
            for idx, state in enumerate(_JOINT_STATES)
        }

        processed_data_dict[pair_key] = {
            "sweep_points": sweep_points,
            **populations,
        }

    return processed_data_dict


@task
def fit_data(
    qubit_pairs: list,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    options: FitDataOptions | None = None,
) -> dict[str, lmfit.model.ModelResult]:
    """Fit an exponential decay to P(00) for each qubit pair.

    Arguments:
        qubit_pairs:
            List of [q0, q1] pairs.
        processed_data_dict:
            Per-pair joint state populations as returned by
            [calculate_qubit_population_tq_rb]().
        options:
            Fit options.

    Returns:
        Dictionary keyed by pair UID with lmfit ModelResult values.
    """
    opts = FitDataOptions() if options is None else options

    fit_results = {}
    if not opts.do_fitting:
        return fit_results

    for pair in qubit_pairs:
        q0, q1 = pair[0], pair[1]
        pair_key = f"{q0.uid}-{q1.uid}"

        if pair_key not in processed_data_dict:
            continue

        swpts_fit = processed_data_dict[pair_key]["sweep_points"]
        data_to_fit = processed_data_dict[pair_key]["00"]

        # P(00) decays from ~1 toward 1/d = 0.25 as the circuit depolarises.
        # Model: A * exp(-decay_rate * m) + offset  with A > 0 (positive amplitude).
        param_hints = {
            "amplitude": {"value": 0.75},
            "decay_rate": {"value": 1 / 50},
            "offset": {"value": 0.25},
        }
        param_hints_user = opts.fit_parameters_hints
        if param_hints_user is None:
            param_hints_user = {}
        param_hints.update(param_hints_user)

        try:
            fit_res = exponential_decay_fit(
                swpts_fit,
                data_to_fit,
                param_hints=param_hints,
            )
            fit_results[pair_key] = fit_res
        except ValueError as err:
            comment(f"Fit failed for {pair_key}: {err}.")

    return fit_results


@task
def compute_pair_epc(
    qubit_pairs: list,
    fit_results: dict[str, lmfit.model.ModelResult],
) -> dict[str, dict[str, float]]:
    """Compute the EPC for each qubit pair from the P(00) fit.

    The error per two-qubit Clifford is extracted from the exponential decay
    of the joint ``"00"`` state population::

        EPC = (d - 1) / d * (1 - exp(-decay_rate))

    with d = 4 (two-qubit Hilbert space dimension).

    Arguments:
        qubit_pairs:
            List of [q0, q1] pairs.
        fit_results:
            Per-pair fit results as returned by [fit_data]().

    Returns:
        Dictionary ``{"q0-q1": {"epc_avg": float}}`` keyed by pair UID.
    """
    epc_factor = (_D_TWO_QUBIT - 1) / _D_TWO_QUBIT  # 3/4 for 2 qubits

    pair_epc = {}
    for pair in qubit_pairs:
        q0, q1 = pair[0], pair[1]
        pair_key = f"{q0.uid}-{q1.uid}"

        if pair_key in fit_results:
            decay_rate = fit_results[pair_key].params["decay_rate"].value
            epc = epc_factor * (1.0 - np.exp(-decay_rate))
        else:
            epc = np.nan

        pair_epc[pair_key] = {"epc_avg": float(epc)}

    return pair_epc


@task_options(base_class=BasePlottingOptions)
class PlotPopulationTQRBOptions:
    """Options for the `plot_population` task of the two-qubit RB analysis.

    Attributes:
        do_fitting:
            Whether to overlay the exponential fit on the P(00) data.
            Default: ``True``.
    """

    do_fitting: bool = option_field(True, description="Whether to overlay the fit.")


@task
def plot_population(
    qubit_pairs: list,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    fit_results: dict[str, lmfit.model.ModelResult] | None,
    options: PlotPopulationTQRBOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Plot the joint two-qubit state populations for each pair.

    One figure is produced per qubit pair, showing P(00) vs. number of
    Cliffords together with the exponential fit and extracted EPC.

    Arguments:
        qubit_pairs:
            List of [q0, q1] pairs.
        processed_data_dict:
            Per-pair joint state populations.
        fit_results:
            Per-pair fit results (keyed by pair UID).
        options:
            Plot options.

    Returns:
        Dictionary keyed by pair UID (``"q0_uid-q1_uid"``) with matplotlib
        Figure objects.
    """
    opts = PlotPopulationTQRBOptions() if options is None else options
    figures = {}

    for pair in qubit_pairs:
        q0, q1 = pair[0], pair[1]
        pair_key = f"{q0.uid}-{q1.uid}"

        if pair_key not in processed_data_dict:
            continue

        data = processed_data_dict[pair_key]
        sweep_points = data["sweep_points"]

        fig, ax = plt.subplots()
        ax.set_title(f"Two-Qubit RB  [{pair_key}]")
        ax.set_xlabel("Number of 2Q Cliffords")
        ax.set_ylabel("P(00)")

        ax.plot(
            sweep_points,
            data["00"],
            "o",
            color="C0",
            zorder=2,
            label="P(00) data",
        )

        # Overlay exponential fit to P(00)
        epc_text = None
        if opts.do_fitting and fit_results and pair_key in fit_results:
            fit_res = fit_results[pair_key]
            swpts_fine = np.linspace(sweep_points.min(), sweep_points.max(), 501)
            ax.plot(
                swpts_fine,
                fit_res.model.func(swpts_fine, **fit_res.best_values),
                "-",
                color="C0",
                zorder=3,
                label="P(00) fit",
            )

            decay_rate_uf = unc.ufloat(
                fit_res.params["decay_rate"].value,
                fit_res.params["decay_rate"].stderr or 0.0,
            )
            fidelity_fit = uexp(-decay_rate_uf)
            epc_factor = (_D_TWO_QUBIT - 1) / _D_TWO_QUBIT
            epc_uf = epc_factor * (1.0 - fidelity_fit)

            epc_text = (
                f"fidelity = {fidelity_fit.nominal_value:.4f}"
                f" ± {fidelity_fit.std_dev:.4f},"
                f"  EPC = {epc_uf.nominal_value:.4f}"
                f" ± {epc_uf.std_dev:.4f}"
            )

        if epc_text:
            ax.text(
                0,
                -0.20,
                epc_text,
                ha="left",
                va="top",
                transform=ax.transAxes,
            )

        ax.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            handlelength=1.5,
            frameon=False,
        )

        if opts.save_figures:
            save_artifact(f"2Q-RB_{pair_key}", fig)

        if opts.close_figures:
            plt.close(fig)

        figures[pair_key] = fig

    return figures


def exponential_decay_fit(
    x: ArrayLike,
    data: ArrayLike,
    param_hints: dict | None = None,
) -> lmfit.model.ModelResult:
    """Fit an exponential decay model to RB data.

    Arguments:
        x: Sequence lengths (number of Cliffords).
        data: Joint state population values.
        param_hints: lmfit parameter hints dictionary.

    Returns:
        The lmfit result.
    """
    if not param_hints:
        param_hints = {
            "decay_rate": {"value": 2 / (3 * np.max(x))},
            "amplitude": {
                "value": abs(np.max(data) - np.min(data)) / 2,
                "min": 0,
            },
            "offset": {"value": 0, "vary": False},
        }

    return fit_data_lmfit(
        fit_mods.exponential_decay,
        x,
        data,
        param_hints=param_hints,
    )
