# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Analysis for the Wigner tomography experiment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from laboneq import workflow
from laboneq.simple import dsl

from laboneq_applications.analysis.options import BasePlottingOptions
from laboneq_applications.conversion import as_quantum_element_list

if TYPE_CHECKING:
    import matplotlib as mpl
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults

    from laboneq_applications.typing import QuantumElements


# ═════════════════════════════════════════════════════════════
# Options
# ═════════════════════════════════════════════════════════════


@workflow.workflow_options
class WignerTomographyAnalysisWorkflowOptions:
    """Options for the Wigner tomography analysis workflow.

    Attributes:
        do_plotting:
            Whether to create plots.
            Default: True.
    """

    do_plotting: bool = workflow.option_field(
        True, description="Whether to create plots."
    )


@workflow.task_options(base_class=BasePlottingOptions)
class PlotWignerFunctionOptions:
    """Options for the Wigner function plotting tasks.

    Attributes:
        w_max_override:
            Override for the Wigner function color scale maximum.
            If None, uses the theoretical maximum 2/π ≈ 0.6366.
            Useful when experimental data has reduced contrast.
            Default: None.
        save_figures:
            Whether to save the figures.
            Default: `True`.
        close_figures:
            Whether to close the figures.
            Default: `True`.
    """

    w_max_override: float | None = workflow.option_field(
        None, description="Override for the Wigner function color scale maximum."
    )


# ═════════════════════════════════════════════════════════════
# Workflow
# ═════════════════════════════════════════════════════════════


@workflow.workflow
def analysis_workflow(
    result: RunExperimentResults,
    qubits: QuantumElements,
    sweep_data: dict,
    options: WignerTomographyAnalysisWorkflowOptions | None = None,
) -> None:
    """The Wigner tomography analysis workflow.

    The workflow consists of the following steps:

    - [calculate_wigner_function][]
    - [plot_wigner_function][]
    - [plot_wigner_line_cuts][]

    For 1D scans, the Wigner function plot is a line plot of W(Re(β))
    together with the intermediate quantities P(|g⟩) and ⟨Π⟩. For 2D scans,
    it is a full phase-space color map, and the line cuts through the origin
    are plotted as well.

    Arguments:
        result:
            The experiment results returned by the run_experiment task.
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the result.
        sweep_data:
            Dictionary returned by ``compute_sweep_parameters``,
            containing the phase-space grid information.
        options:
            The options for building the workflow, passed as an instance of
            [WignerTomographyAnalysisWorkflowOptions][].

    Returns:
        WorkflowBuilder:
            The builder for the analysis workflow. Its output is the
            dictionary returned by [calculate_wigner_function][].
    """
    wigner_data = calculate_wigner_function(qubits, result, sweep_data)
    with workflow.if_(options.do_plotting):
        plot_wigner_function(qubits, wigner_data)
        plot_wigner_line_cuts(qubits, wigner_data)
    workflow.return_(wigner_data)


# ═════════════════════════════════════════════════════════════
# Analysis tasks
# ═════════════════════════════════════════════════════════════


@workflow.task
def calculate_wigner_function(
    qubits: QuantumElements,
    result: RunExperimentResults,
    sweep_data: dict,
) -> dict[str, dict]:
    """Calculate the Wigner function from the averaged discrimination results.

    Arguments:
        qubits:
            The qubits on which to run the analysis.
        result:
            The experiment results returned by the run_experiment task.
        sweep_data:
            Dictionary returned by ``compute_sweep_parameters``.

    Returns:
        Dictionary keyed by qubit UID with the Wigner function ``"w"``, the
            parity ``"parity"``, the ground-state probability ``"pg"``, the scan
            axes ``"beta_re"`` and ``"beta_im"``, and the flag ``"is_2d"``.
    """
    qubits = as_quantum_element_list(qubits)

    wigner_data = {}
    for i, q in enumerate(qubits):
        handle = dsl.handles.result_handle(q.uid)
        data_raw = result.get_data(handle)

        # Slice out this qubit's per-qubit sweep_data entries.
        qubit_sweep_data = _qubit_sweep_data(sweep_data, i)

        wigner_data[q.uid] = _analyze_single_qubit(
            q,
            data_raw,
            qubit_sweep_data,
        )

    return wigner_data


@workflow.task
def plot_wigner_function(
    qubits: QuantumElements,
    wigner_data: dict[str, dict],
    options: PlotWignerFunctionOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Plot the Wigner function: a line plot for 1D scans, a map for 2D scans.

    Arguments:
        qubits:
            The qubits on which to run the analysis.
        wigner_data:
            The dictionary returned by calculate_wigner_function.
        options:
            The options for this task as an instance of [PlotWignerFunctionOptions][].

    Returns:
        dict with qubit UIDs as keys and the figures for each qubit as values.
    """
    opts = PlotWignerFunctionOptions() if options is None else options
    qubits = as_quantum_element_list(qubits)
    w_max = opts.w_max_override or (2.0 / np.pi)
    figures = {}
    for q in qubits:
        qubit_result = wigner_data[q.uid]
        if qubit_result["is_2d"]:
            fig = _plot_wigner_2d(
                q.uid,
                qubit_result["w"],
                qubit_result["beta_re"],
                qubit_result["beta_im"],
                w_max=w_max,
            )
            name = "wigner_2d"
        else:
            fig = _plot_wigner_1d(
                q.uid,
                qubit_result["w"],
                qubit_result["parity"],
                qubit_result["pg"],
                qubit_result["beta_re"],
                w_max=w_max,
            )
            name = "wigner_1d"
        _save_and_close(fig, f"wigner_{q.uid}_{name}", opts)
        figures[q.uid] = fig
    return figures


@workflow.task
def plot_wigner_line_cuts(
    qubits: QuantumElements,
    wigner_data: dict[str, dict],
    options: PlotWignerFunctionOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Plot line cuts through the origin of 2D Wigner function scans.

    Qubits with a 1D scan are skipped.

    Arguments:
        qubits:
            The qubits on which to run the analysis.
        wigner_data:
            The dictionary returned by calculate_wigner_function.
        options:
            The options for this task as an instance of [PlotWignerFunctionOptions][].

    Returns:
        dict with qubit UIDs as keys and the figures for each qubit with a 2D
            scan as values.
    """
    opts = PlotWignerFunctionOptions() if options is None else options
    qubits = as_quantum_element_list(qubits)
    w_max = opts.w_max_override or (2.0 / np.pi)
    figures = {}
    for q in qubits:
        qubit_result = wigner_data[q.uid]
        if not qubit_result["is_2d"]:
            continue
        fig = _plot_wigner_line_cuts(
            q.uid,
            qubit_result["w"],
            qubit_result["beta_re"],
            qubit_result["beta_im"],
            w_max=w_max,
        )
        _save_and_close(fig, f"wigner_{q.uid}_wigner_cuts", opts)
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


def _qubit_sweep_data(sweep_data: dict, i: int) -> dict:
    """Extract the i-th qubit's sweep entries.

    Backward-compatible: if a key happens to hold a scalar
    (legacy single-qubit ``compute_sweep_parameters`` output),
    it is returned unchanged. Otherwise the i-th element of the
    list is returned.
    """

    def pick(value: object) -> object:
        # Per-qubit list -> select element i.
        # Scalar/np.ndarray -> assume legacy single-qubit layout.
        if isinstance(value, list):
            return value[i]
        return value

    return {
        "amp_values": pick(sweep_data["amp_values"]),
        "phase_values": pick(sweep_data["phase_values"]),
        "beta_re_values": pick(sweep_data["beta_re_values"]),
        "beta_im_values": pick(sweep_data["beta_im_values"]),
        "is_2d": pick(sweep_data["is_2d"]),
        "n_re": pick(sweep_data["n_re"]),
        "n_im": pick(sweep_data["n_im"]),
    }


# ═════════════════════════════════════════════════════════════
# Single-qubit analysis
# ═════════════════════════════════════════════════════════════


def _analyze_single_qubit(
    q: QuantumElements,
    data_raw: np.ndarray,
    sweep_data: dict,
) -> dict:
    """Run Wigner function extraction for a single qubit.

    ``sweep_data`` here is the *per-qubit* dict produced by
    ``_qubit_sweep_data``; its scalar entries (``is_2d``, ``n_re``,
    ``n_im``) and array entries (``beta_re_values``,
    ``beta_im_values``) belong to a single qubit.
    """
    raw = np.real(np.asarray(data_raw))

    # Discrimination convention: averaged result ≈ P(|e⟩)
    pg = 1.0 - raw
    parity = 2.0 * pg - 1.0
    w = (2.0 / np.pi) * parity

    is_2d = sweep_data["is_2d"]
    if is_2d:
        n_re = sweep_data["n_re"]
        n_im = sweep_data["n_im"]
        w = w.reshape(n_re, n_im)
        parity = parity.reshape(n_re, n_im)
        pg = pg.reshape(n_re, n_im)

    return {
        "w": w,
        "parity": parity,
        "pg": pg,
        "beta_re": sweep_data["beta_re_values"],
        "beta_im": sweep_data["beta_im_values"],
        "is_2d": is_2d,
    }


# ═════════════════════════════════════════════════════════════
# Plotting helpers
# ═════════════════════════════════════════════════════════════


def _plot_wigner_2d(
    qubit_uid: str,
    w: np.ndarray,
    beta_re: np.ndarray,
    beta_im: np.ndarray,
    *,
    w_max: float = 2.0 / np.pi,
) -> plt.Figure:
    """Plot the 2D Wigner function as a phase-space color map.

    Arguments:
        qubit_uid:
            Qubit UID for the plot title.
        w:
            2D Wigner function array, shape ``(n_re, n_im)``.
        beta_re:
            Re(β) values.
        beta_im:
            Im(β) values.
        w_max:
            Symmetric color scale limit. Default: 2/π.

    Returns:
        fig:
            The matplotlib figure.
    """
    fig, ax = plt.subplots(
        figsize=(7, 6),
        constrained_layout=True,
    )

    pcm = ax.pcolormesh(
        beta_re,
        beta_im,
        w.T,
        cmap="RdBu_r",
        vmin=-w_max,
        vmax=w_max,
        shading="auto",
    )
    ax.set_xlabel(r"Re($\beta$)")
    ax.set_ylabel(r"Im($\beta$)")
    ax.set_title(f"{qubit_uid} — Wigner Function $W(\\beta)$")
    ax.set_aspect("equal")
    fig.colorbar(pcm, ax=ax, label=r"$W(\beta)$", shrink=0.8)

    return fig


def _plot_wigner_line_cuts(
    qubit_uid: str,
    w: np.ndarray,
    beta_re: np.ndarray,
    beta_im: np.ndarray,
    *,
    w_max: float = 2.0 / np.pi,
) -> plt.Figure:
    """Plot Wigner function line cuts through the origin.

    Produces two panels:

    - Left: cut along Re(β) at Im(β) ≈ 0.
    - Right: cut along Im(β) at Re(β) ≈ 0.

    Arguments:
        qubit_uid:
            Qubit UID for the plot title.
        w:
            2D Wigner function array, shape ``(n_re, n_im)``.
        beta_re:
            Re(β) values.
        beta_im:
            Im(β) values.
        w_max:
            Y-axis limit. Default: 2/π.

    Returns:
        fig:
            The matplotlib figure.
    """
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
        constrained_layout=True,
    )

    # Cut along Re(β) at Im(β) ≈ 0
    idx_im0 = np.argmin(np.abs(beta_im))
    ax = axes[0]
    ax.plot(beta_re, w[:, idx_im0], "b-o", ms=3)
    ax.fill_between(
        beta_re,
        w[:, idx_im0],
        alpha=0.15,
        color="blue",
    )
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.set_xlabel(r"Re($\beta$)")
    ax.set_ylabel(r"$W(\beta)$")
    ax.set_ylim(-w_max * 1.15, w_max * 1.15)
    ax.set_title(rf"Im($\beta$) = {beta_im[idx_im0]:.2f}")

    # Cut along Im(β) at Re(β) ≈ 0
    idx_re0 = np.argmin(np.abs(beta_re))
    ax = axes[1]
    ax.plot(beta_im, w[idx_re0, :], "r-s", ms=3)
    ax.fill_between(
        beta_im,
        w[idx_re0, :],
        alpha=0.15,
        color="red",
    )
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.set_xlabel(r"Im($\beta$)")
    ax.set_ylabel(r"$W(\beta)$")
    ax.set_ylim(-w_max * 1.15, w_max * 1.15)
    ax.set_title(rf"Re($\beta$) = {beta_re[idx_re0]:.2f}")

    fig.suptitle(
        f"{qubit_uid} — Wigner Function Line Cuts",
        fontsize=13,
    )

    return fig


def _plot_wigner_1d(
    qubit_uid: str,
    w: np.ndarray,
    parity: np.ndarray,
    pg: np.ndarray,
    beta_re: np.ndarray,
    *,
    w_max: float = 2.0 / np.pi,
) -> plt.Figure:
    """Plot the 1D Wigner function scan with intermediate quantities.

    Produces two panels:

    - Left: Wigner function W(β) vs Re(β).
    - Right: P(|g⟩) and ⟨Π⟩ vs Re(β).

    Arguments:
        qubit_uid:
            Qubit UID for the plot title.
        w:
            1D Wigner function array, shape ``(n_re,)``.
        parity:
            1D parity expectation values, shape ``(n_re,)``.
        pg:
            1D ground-state probabilities, shape ``(n_re,)``.
        beta_re:
            Re(β) values.
        w_max:
            Y-axis limit for the Wigner panel. Default: 2/π.

    Returns:
        fig:
            The matplotlib figure.
    """
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
        constrained_layout=True,
    )

    # Left panel: Wigner function
    ax = axes[0]
    ax.plot(
        beta_re,
        w,
        "b-o",
        ms=4,
        label=r"$W(\beta)$",
    )
    ax.fill_between(
        beta_re,
        w,
        alpha=0.15,
        color="blue",
    )
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.set_xlabel(r"Re($\beta$)")
    ax.set_ylabel(r"$W(\beta)$")
    ax.set_ylim(-w_max * 1.15, w_max * 1.15)
    ax.set_title("Wigner Function")
    ax.legend()

    # Right panel: P(g) and parity
    ax = axes[1]
    ax.plot(
        beta_re,
        pg,
        "g-^",
        ms=4,
        label=r"$P(|g\rangle)$",
    )
    ax.plot(
        beta_re,
        parity,
        "m-v",
        ms=4,
        label=r"$\langle\hat{\Pi}\rangle$",
    )
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.axhline(1, color="gray", ls=":", lw=0.5)
    ax.set_xlabel(r"Re($\beta$)")
    ax.set_ylabel("Value")
    ax.set_ylim(-1.15, 1.15)
    ax.set_title("Transmon Occupation & Parity")
    ax.legend()

    fig.suptitle(
        f"{qubit_uid} — Wigner Tomography "
        r"[Im($\beta$) = 0]",
        fontsize=13,
    )

    return fig
