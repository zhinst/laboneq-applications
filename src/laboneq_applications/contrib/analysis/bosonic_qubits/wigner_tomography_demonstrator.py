# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Analysis for the Wigner tomography experiment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from laboneq import workflow
from laboneq.simple import dsl

if TYPE_CHECKING:
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults

    from laboneq_applications.typing import QuantumElements


# ═════════════════════════════════════════════════════════════
# Options
# ═════════════════════════════════════════════════════════════


@workflow.task_options
class WignerTomographyAnalysisOptions:
    """Options for the Wigner tomography analysis.

    Attributes:
        generate_plot:
            Whether to generate plots. Default: True.
        show_plot:
            Whether to show plots in the terminal. Default: True.
        save_figures:
            Whether to save the figures. Default: False.
        save_path:
            Path to save the figures. Default: "./"
        w_max_override:
            Override for the Wigner function color scale maximum.
            If None, uses the theoretical maximum 2/π ≈ 0.6366.
            Useful when experimental data has reduced contrast.
            Default: None.
    """

    generate_plot: bool = True
    show_plot: bool = True
    save_figures: bool = False
    save_path: str = "./"
    w_max_override: float | None = None


@workflow.workflow_options
class WignerTomographyAnalysisWorkflowOptions:
    """Options for the Wigner tomography analysis workflow."""


# ═════════════════════════════════════════════════════════════
# Workflow
# ═════════════════════════════════════════════════════════════


@workflow.workflow(name="wigner_tomography_analysis")
def analysis_workflow(
    result: RunExperimentResults,
    qpu: QPU,
    qubits: QuantumElements,
    sweep_data: dict,
    *,
    options: WignerTomographyAnalysisWorkflowOptions | None = None,
) -> None:
    """Wigner tomography analysis workflow.

    Extracts the Wigner function from the raw measurement data
    and produces plots for each qubit.

    For 1D scans, the output includes a line plot of W(Re(β))
    together with the intermediate quantities P(|g⟩) and ⟨Π⟩.

    For 2D scans, the output includes a full phase-space color
    map and line cuts through the origin.

    Arguments:
        result:
            The result of the Wigner tomography experiment.
        qpu:
            The qpu consisting of the original qubits and
            quantum operations.
        qubits:
            The qubits the experiment was run on.
        sweep_data:
            Dictionary returned by ``compute_sweep_parameters``,
            containing the phase-space grid information.
        options:
            The options for the analysis workflow.

    Returns:
        result:
            The result of the analysis workflow.
    """
    analyze(
        result,
        qpu,
        qubits,
        sweep_data=sweep_data,
    )


# ═════════════════════════════════════════════════════════════
# Analysis task
# ═════════════════════════════════════════════════════════════


@workflow.task
def analyze(
    result: RunExperimentResults,
    qpu: QPU,
    qubits: QuantumElements,
    sweep_data: dict,
    options: WignerTomographyAnalysisOptions | None = None,
) -> dict:
    """Analyze the Wigner tomography results."""
    opts = WignerTomographyAnalysisOptions() if options is None else options

    qubits = _ensure_list(qubits)

    analysis_results = {}
    for i, q in enumerate(qubits):
        handle = dsl.handles.result_handle(q.uid)
        data_raw = result.get_data(handle)

        # Slice out this qubit's per-qubit sweep_data entries.
        qubit_sweep_data = _qubit_sweep_data(sweep_data, i)

        qubit_result = _analyze_single_qubit(
            q,
            data_raw,
            qubit_sweep_data,
        )

        if opts.generate_plot:
            _generate_plots(qubit_result, q.uid, opts)

        analysis_results[q.uid] = qubit_result

    return analysis_results


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


def _ensure_list(qubits: QuantumElements) -> list:
    if not isinstance(qubits, list):
        return [qubits]
    return qubits


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
        "figures": {},
    }


# ═════════════════════════════════════════════════════════════
# Plot generation
# ═════════════════════════════════════════════════════════════


def _generate_plots(
    qubit_result: dict,
    q_uid: str,
    opts: WignerTomographyAnalysisOptions,
) -> None:
    """Generate all Wigner tomography plots for a single qubit.

    Dispatches to the appropriate plotting functions based on
    whether the scan is 1D or 2D, and handles figure saving
    and display.

    Arguments:
        qubit_result:
            Analysis result dictionary for a single qubit.
            The ``"figures"`` sub-dictionary is populated
            in-place.
        q_uid:
            Qubit UID for plot titles and filenames.
        opts:
            Analysis options controlling plot generation,
            saving, and display.
    """
    is_2d = qubit_result["is_2d"]
    w_max = opts.w_max_override or (2.0 / np.pi)

    if is_2d:
        qubit_result["figures"]["wigner_2d"] = _plot_wigner_2d(
            q_uid,
            qubit_result["w"],
            qubit_result["beta_re"],
            qubit_result["beta_im"],
            w_max=w_max,
        )
        qubit_result["figures"]["wigner_cuts"] = _plot_wigner_line_cuts(
            q_uid,
            qubit_result["w"],
            qubit_result["beta_re"],
            qubit_result["beta_im"],
            w_max=w_max,
        )
    else:
        qubit_result["figures"]["wigner_1d"] = _plot_wigner_1d(
            q_uid,
            qubit_result["w"],
            qubit_result["parity"],
            qubit_result["pg"],
            qubit_result["beta_re"],
            w_max=w_max,
        )

    if opts.save_figures:
        for name, fig in qubit_result["figures"].items():
            fig.savefig(
                f"{opts.save_path}/wigner_{q_uid}_{name}.png",
                dpi=150,
                bbox_inches="tight",
            )
    if opts.show_plot:
        plt.show()


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
