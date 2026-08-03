# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Wigner Tomography Demonstrator Workflow.

Showcases the ``wigner_point`` quantum operation by running a
Wigner tomography scan on a bosonic qubit's memory cavity.

Supports both 1D line cuts (when ``beta_im_values`` is ``None``)
and full 2D phase-space scans (when both ``beta_re_values`` and
``beta_im_values`` are provided).

Pulse sequence per sweep point::

    D(-beta) → [parity mapping] → Readout → Passive Reset

The Wigner function at each phase-space point β is recovered as:

    W(beta) = (2/π) * [2 * P(|g⟩) - 1]

Example — 1D scan along the real axis::

    wigner_tomography_demonstrator(
        session=session,
        qop=qop,
        q=qubit,
        beta_re_values=np.linspace(-4, 4, 41),
    )

Example — full 2D scan::

    wigner_tomography_demonstrator(
        session=session,
        qop=qop,
        q=qubit,
        beta_re_values=np.linspace(-4, 4, 41),
        beta_im_values=np.linspace(-4, 4, 41),
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from laboneq import workflow
from laboneq.dsl.quantum import QuantumParameters
from laboneq.simple import (
    AcquisitionType,
    AveragingMode,
    Experiment,
    SweepParameter,
    dsl,
)
from laboneq.workflow.tasks import (
    compile_experiment,
    run_experiment,
)

from laboneq_applications.contrib.analysis.bosonic_qubits import (
    wigner_tomography_demonstrator as wigner_tomography_demonstrator_analysis,
)
from laboneq_applications.experiments.options import (
    BaseExperimentOptions,
)
from laboneq_applications.tasks import (
    temporary_qpu,
    temporary_quantum_elements_from_qpu,
)

if TYPE_CHECKING:
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.session import Session

    from laboneq_applications.typing import QuantumElements


# ═════════════════════════════════════════════════════════════
# Options
# ═════════════════════════════════════════════════════════════


@workflow.task_options(base_class=BaseExperimentOptions)
class WignerTomographyExperimentOptions:
    """Options for the Wigner tomography demonstrator experiment.

    Attributes:
        averaging_mode:
            LabOne Q averaging mode. Default: ``CYCLIC``.
        acquisition_type:
            LabOne Q acquisition type. Default:
            ``DISCRIMINATION``. Requires calibrated
            discrimination thresholds on the readout acquire
            line.
        chunk_count:
            Number of sweep points per chunk. Default: 0 (auto
            chunking). Adjusting this can help manage memory usage
            and experiment duration, especially for large 2D scans.
    """

    averaging_mode: AveragingMode = AveragingMode.CYCLIC
    acquisition_type: AcquisitionType = AcquisitionType.DISCRIMINATION
    chunk_count: int = 1  # 0 = auto chunking


@workflow.workflow_options
class WignerTomographyWorkflowOptions:
    """Options for the Wigner tomography demonstrator workflow.

    Attributes:
        do_analysis (bool):
            The option for performing the analysis.
    """

    do_analysis: bool = True


# ═════════════════════════════════════════════════════════════
# Workflows
# ═════════════════════════════════════════════════════════════


@workflow.workflow(name="wigner_tomography_demonstrator")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: list[str] | str,
    *,
    beta_re_values: list[np.ndarray],
    beta_im_values: list[np.ndarray | None] | None = None,
    temporary_parameters: dict[str, dict | QuantumParameters] | None = None,
    options: WignerTomographyWorkflowOptions | None = None,
) -> None:
    """Wigner tomography demonstrator workflow.

    The workflow consists of the following steps:

    - [compute_sweep_parameters]()
    - [create_experiment]()
    - [compile_experiment]()
    - [run_experiment]()
    - [analysis_workflow]()

    Steps that can be added in the future include:

    - [update_qpu]()

    The workflow supports two scan modes:

    **1D line cut** (``beta_im_values=None``):
        Scans along the real axis with Im(beta) = 0.
        Useful for quick diagnostics and state verification.

    **2D phase-space scan** (both arrays provided):
        Scans a full Re(beta) * Im(beta) grid.
        Produces a complete Wigner function tomogram.

    Arguments:
        session:
            Connected LabOne Q session.
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubits to run the experiments on.
        beta_re_values:
            1D array of Re(β) values to scan.
        beta_im_values:
            1D array of Im(β) values, or ``None`` for a
            1D scan along the real axis. Default: ``None``.
        temporary_parameters:
            The temporary parameters to update the qubits with.
        options:
            Workflow options. If ``None``, default options are
            used. Default: ``None``.

    Returns:
        Tuple of ``(result, wigner_data, figures)`` where:

        ``result``
            The raw LabOne Q experiment results.
        ``wigner_data``
            Dictionary with the extracted Wigner function,
            parity, and P(|g⟩) values.
        ``figures``
            Dictionary mapping figure names to matplotlib
            ``Figure`` objects.

    Example:
        1D line cut along the real axis:

        ```python
        # QPU from a single-qubit device setup
        qpu = QPU(
            quantum_elements=BosonicQubit.from_device_setup(setup),
            quantum_operations=BosonicQubitOperations(),
        )
        result = experiment_workflow(
            session=session,
            qpu=qpu,
            qubits="q0",
            beta_re_values=[np.linspace(-4, 4, 41)],
        ).run()
        ```

        Full 2D phase-space scan:

        ```python
        # QPU from a single-qubit device setup
        qpu = QPU(
            quantum_elements=BosonicQubit.from_device_setup(setup),
            quantum_operations=BosonicQubitOperations(),
        )
        result = experiment_workflow(
            session=session,
            qpu=qpu,
            qubits="q0",
            beta_re_values=[np.linspace(-4, 4, 41)],
            beta_im_values=[np.linspace(-4, 4, 41)],
        ).run()
        ```
    """
    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    qubits = temporary_quantum_elements_from_qpu(temp_qpu, qubits)
    sweep_data = compute_sweep_parameters(beta_re_values, beta_im_values)
    exp = create_experiment(
        qpu,
        qubits,
        sweep_data,
    )
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)
    with workflow.if_(options.do_analysis):
        wigner_tomography_demonstrator_analysis.analysis_workflow(
            result, qpu, qubits, sweep_data
        )
    workflow.return_(result)


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubits: QuantumElements,
    sweep_data: dict,
    options: WignerTomographyExperimentOptions | None = None,
) -> Experiment:
    """Create the Wigner tomography LabOne Q experiment.

    Builds an experiment that sweeps over phase-space points β for
    *all* qubits in lockstep and, at each point, displaces each
    cavity by -β, maps the photon-number parity onto the transmon
    state, and reads out the transmon.

    Arguments:
        qpu:
            The qpu consisting of the original qubits and quantum
            operations.
        qubits:
            The bosonic qubits to measure in parallel.
        sweep_data:
            Dictionary returned by ``compute_sweep_parameters``.
            Must provide per-qubit ``amp_values`` and
            ``phase_values`` lists, all of equal length.
        options:
            Workflow options.

    Returns:
        The LabOne Q ``Experiment`` object, ready for compilation.
    """
    opts = WignerTomographyExperimentOptions() if options is None else options

    qop = qpu.quantum_operations

    # One SweepParameter per qubit per axis (|beta|, arg(beta)).
    # All arrays must share the same length so they can be swept
    # simultaneously.
    amp_params = [
        SweepParameter(
            uid=f"beta_amp_{q.uid}",
            values=sweep_data["amp_values"][i],
        )
        for i, q in enumerate(qubits)
    ]
    phase_params = [
        SweepParameter(
            uid=f"beta_phase_{q.uid}",
            values=sweep_data["phase_values"][i],
        )
        for i, q in enumerate(qubits)
    ]

    sweep_kwargs = {
        "uid": "wigner_sweep",
        # All amp and phase parameters advance in lockstep.
        "parameter": amp_params + phase_params,
    }
    if opts.chunk_count > 0:
        sweep_kwargs["chunk_count"] = opts.chunk_count
    else:
        sweep_kwargs["auto_chunking"] = True

    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
    ):
        with dsl.sweep(**sweep_kwargs):
            for q, amp_param, phase_param in zip(
                qubits,
                amp_params,
                phase_params,
                strict=False,
            ):
                qop.wigner_point(
                    q,
                    beta_amplitude=amp_param,
                    beta_phase=phase_param,
                    handle=dsl.handles.result_handle(q.uid),
                )
                qop.passive_reset(
                    q,
                    delay=q.parameters.reset_delay_length,
                )


@workflow.task
def compute_sweep_parameters(
    beta_re_values: list[np.ndarray] | list[list[float]],
    beta_im_values: (list[np.ndarray | None] | list[list[float] | None] | None),
) -> dict:
    """Convert per-qubit Cartesian β values to polar sweep arrays.

    Each qubit gets its own Re(β) (and optionally Im(β)) list. For a
    1D scan, pass ``beta_im_values=None`` (or a list whose entries
    are all ``None``). For a 2D scan, pass an Im(β) array per qubit.

    All qubits must end up with the **same number of flattened sweep
    points**, so they can be swept simultaneously in a single
    ``dsl.sweep``. The simplest way to ensure this is to use the
    same grid shape for every qubit; the actual values may differ.

    Arguments:
        beta_re_values:
            List of 1D arrays of Re(β) values, one per qubit.
        beta_im_values:
            List of 1D arrays of Im(β) values (one per qubit), or
            ``None`` for a 1D scan along the real axis. Individual
            entries may also be ``None`` to mix 1D and 2D scans, as
            long as the flattened length is consistent.

    Returns:
        Dictionary containing per-qubit (list-indexed) entries:

        ``"amp_values"``
            List of flattened |β| arrays, one per qubit.
        ``"phase_values"``
            List of flattened arg(β) arrays (radians), one per
            qubit.
        ``"beta_re_values"``
            List of input Re(β) arrays.
        ``"beta_im_values"``
            List of input Im(β) arrays (or ``None`` per qubit).
        ``"is_2d"``
            List of bool flags, ``True`` if that qubit's scan is 2D.
        ``"n_re"``, ``"n_im"``
            Lists of grid sizes per qubit.
    """
    n_qubits = len(beta_re_values)

    # Normalise beta_im_values into a per-qubit list of (array | None).
    if beta_im_values is None:
        beta_im_per_qubit = [None] * n_qubits
    else:
        beta_im_per_qubit = list(beta_im_values)
        if len(beta_im_per_qubit) != n_qubits:
            raise ValueError(
                "beta_im_values must have one entry per qubit "
                f"(got {len(beta_im_per_qubit)}, expected {n_qubits})."
            )

    amp_values_list: list[np.ndarray] = []
    phase_values_list: list[np.ndarray] = []
    beta_re_list: list[np.ndarray] = []
    beta_im_list: list[np.ndarray | None] = []
    is_2d_list: list[bool] = []
    n_re_list: list[int] = []
    n_im_list: list[int] = []

    for q_re, q_im in zip(beta_re_values, beta_im_per_qubit, strict=False):
        beta_re = np.asarray(q_re, dtype=float)

        if q_im is None:
            # 1D scan along the real axis for this qubit.
            re_flat = beta_re
            im_flat = np.zeros_like(beta_re)
            is_2d = False
            n_re = len(beta_re)
            n_im = 1
            beta_im_out: np.ndarray | None = None
        else:
            beta_im = np.asarray(q_im, dtype=float)
            re, im = np.meshgrid(beta_re, beta_im, indexing="ij")
            re_flat = re.ravel()
            im_flat = im.ravel()
            is_2d = True
            n_re = len(beta_re)
            n_im = len(beta_im)
            beta_im_out = beta_im

        amp = np.sqrt(re_flat**2 + im_flat**2)
        phase = np.arctan2(im_flat, re_flat)
        phase[amp == 0.0] = 0.0  # undefined phase at the origin

        amp_values_list.append(amp)
        phase_values_list.append(phase)
        beta_re_list.append(beta_re)
        beta_im_list.append(beta_im_out)
        is_2d_list.append(is_2d)
        n_re_list.append(n_re)
        n_im_list.append(n_im)

    # Hard requirement for simultaneous sweep: all qubits must
    # contribute the same number of sweep points.
    lengths = {len(a) for a in amp_values_list}
    if len(lengths) != 1:
        raise ValueError(
            "All qubits must produce the same number of flattened "
            f"sweep points for a simultaneous sweep; got {lengths}. "
            "Use grids of the same shape for every qubit."
        )

    return {
        "amp_values": amp_values_list,
        "phase_values": phase_values_list,
        "beta_re_values": beta_re_list,
        "beta_im_values": beta_im_list,
        "is_2d": is_2d_list,
        "n_re": n_re_list,
        "n_im": n_im_list,
    }
