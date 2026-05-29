# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""SWAP gate calibration workflow.

The SWAP gate exchanges a single excitation between the ancilla
transmon and the memory cavity:

    |e, n⟩  ↔  |g, n+1⟩

Combined with a preceding transmon π pulse, it prepares a
|1⟩ Fock state in the cavity starting from vacuum:

    |g, 0⟩ --[π]--> |e, 0⟩ --[SWAP]--> |g, 1⟩

The SWAP is parameterised by the drive amplitude and duration.

Calibration modes:
1. Length sweep:
   - SWAP length swept in near time
   - amplitude fixed from stored qubit parameter

2. Amplitude sweep:
   - SWAP amplitude swept in real time
   - length fixed from stored qubit parameter

3. 2D sweep:
   - length swept in near time
   - amplitude swept in real time

Pulse sequence per qubit (all modes):

    qb --- [Rx(π)] --- [SWAP(amp, length)] --- [measure transmon]

If multiple qubits are passed, pulse sequences are applied in
parallel on all qubits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from laboneq import workflow
from laboneq.dsl.quantum import QuantumElement, QuantumOperations, QuantumParameters
from laboneq.simple import Experiment, SweepParameter, dsl
from laboneq.workflow.tasks import (
    compile_experiment,
    run_experiment,
)

from laboneq_applications.contrib.analysis.bosonic_qubits import (
    swap_calibration as swap_calibration_analysis,
)
from laboneq_applications.core import validation
from laboneq_applications.experiments.options import (
    TuneupExperimentOptions,
)
from laboneq_applications.tasks import (
    temporary_qpu,
    temporary_quantum_elements_from_qpu,
)

if TYPE_CHECKING:
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.session import Session

    from laboneq_applications.typing import (
        QuantumElements,
        QubitSweepPoints,
    )


@workflow.task_options(base_class=TuneupExperimentOptions)
class SwapExperimentOptions:
    """Options for the SWAP calibration experiment."""

    swap_pulse: dict | None = None
    chunk_count: int = 1


@workflow.workflow_options
class SwapCalibrationWorkflowOptions:
    """Options for the SWAP calibration workflow.

    Attributes:
        do_analysis:
            Whether to run the analysis workflow after the
            experiment. Default: True.
    """

    do_analysis: bool = True


@workflow.workflow(name="swap_calibration")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: list[str] | str,
    *,
    swap_durations: QubitSweepPoints | None = None,
    swap_amplitudes: QubitSweepPoints | None = None,
    temporary_parameters: dict[str, dict | QuantumParameters] | None = None,
    options: SwapCalibrationWorkflowOptions | None = None,
) -> None:
    """The SWAP gate calibration workflow.

    The workflow consists of the following steps:

    - [create_experiment]()
    - [compile_experiment]()
    - [run_experiment]()
    - [analysis_workflow]()

    Steps that can be added in the future include:

    - [update_qpu]()

    Modes:
        - Only swap_durations given  -> near-time length sweep
        - Only swap_amplitudes given -> real-time amplitude sweep
        - Both given                 -> 2D sweep with near-time
                                        length and real-time amplitude

    Arguments:
        session:
            The connected session to use for running the experiment.
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubits to run the experiments on, passed by UID. May be
            either a single qubit or a list of qubits.
        swap_durations:
            The SWAP pulse durations to sweep for each qubit (near-time
            sweep). If ``None``, no length sweep is performed. At least one
            of ``swap_durations`` or ``swap_amplitudes`` must be provided.
        swap_amplitudes:
            The SWAP pulse amplitudes to sweep for each qubit (real-time
            sweep). If ``None``, no amplitude sweep is performed. At least
            one of ``swap_durations`` or ``swap_amplitudes`` must be
            provided.
        temporary_parameters:
            The temporary parameters to update the qubits with.
        options:
            The options for building the workflow.
            In addition to options from [WorkflowOptions], the following
            custom options are supported:
                - create_experiment: The options for creating the experiment.

    Returns:
        result:
            The result of the workflow.

    Raises:
        ValueError:
            If both ``swap_durations`` and ``swap_amplitudes`` are ``None``.

    Example:
        Length sweep:

        ```python
        result = swap_calibration.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=qpu.quantum_element_uids,
            swap_durations=[np.linspace(100e-9, 600e-9, 51)],
        ).run()
        ```

        Amplitude sweep:

        ```python
        result = swap_calibration.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=qpu.quantum_element_uids,
            swap_amplitudes=[np.linspace(0.1, 0.9, 51)],
        ).run()
        ```

        2D sweep (length x amplitude):

        ```python
        result = swap_calibration.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=qpu.quantum_element_uids,
            swap_durations=[np.linspace(100e-9, 600e-9, 21)],
            swap_amplitudes=[np.linspace(0.1, 0.9, 21)],
        ).run()
        ```
    """
    if swap_durations is None and swap_amplitudes is None:
        raise ValueError(
            "At least one of 'swap_durations' or 'swap_amplitudes' must be provided."
        )

    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    qubits = temporary_quantum_elements_from_qpu(temp_qpu, qubits)

    exp = create_experiment(
        temp_qpu,
        qubits,
        swap_durations=swap_durations,
        swap_amplitudes=swap_amplitudes,
    )
    compiled = compile_experiment(session, exp)
    result = run_experiment(session, compiled)
    with workflow.if_(options.do_analysis):
        swap_calibration_analysis.analysis_workflow(
            result,
            qpu,
            qubits,
            swap_durations=swap_durations,
            swap_amplitudes=swap_amplitudes,
        )
    workflow.return_(result)


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubits: QuantumElements,
    swap_durations: QubitSweepPoints | None = None,
    swap_amplitudes: QubitSweepPoints | None = None,
    options: SwapExperimentOptions | None = None,
) -> Experiment:
    """Creates the SWAP calibration experiment.

    Sweep placement:
        - 1D length sweep    -> near time
        - 1D amplitude sweep -> real time
        - 2D sweep           -> length near time, amplitude real time
    """
    opts = SwapExperimentOptions() if options is None else options
    sweep_mode = _get_sweep_mode(swap_durations, swap_amplitudes)

    if swap_durations is not None:
        qubits, swap_durations = validation.validate_and_convert_qubits_sweeps(
            qubits, swap_durations
        )
    if swap_amplitudes is not None:
        qubits, swap_amplitudes = validation.validate_and_convert_qubits_sweeps(
            qubits, swap_amplitudes
        )

    n_qubits = len(qubits)
    q_durations_list = (
        swap_durations if swap_durations is not None else [None] * n_qubits
    )
    q_amplitudes_list = (
        swap_amplitudes if swap_amplitudes is not None else [None] * n_qubits
    )

    qop = qpu.quantum_operations
    rt_kwargs = _acquire_loop_rt_kwargs(opts)

    if sweep_mode == "length":
        length_params = _make_parallel_sweep_parameters(
            qubits,
            q_durations_list,
            "swap_duration",
        )
        with dsl.sweep(
            uid="swap_cal_len_nt",
            parameter=length_params,
        ):
            with dsl.acquire_loop_rt(**rt_kwargs):
                for q, length_param in zip(qubits, length_params, strict=False):
                    _play_swap_sequence(
                        q=q,
                        qop=qop,
                        opts=opts,
                        length=length_param,
                    )

    elif sweep_mode == "amplitude":
        with dsl.acquire_loop_rt(**rt_kwargs):
            for q, q_amps in zip(qubits, q_amplitudes_list, strict=False):
                with dsl.sweep(
                    **_sweep_kwargs_1d(
                        uid=f"swap_cal_amp_{q.uid}",
                        param_name=f"swap_amplitude_{q.uid}",
                        values=q_amps,
                        opts=opts,
                    )
                ) as amplitude:
                    _play_swap_sequence(
                        q=q,
                        qop=qop,
                        opts=opts,
                        amplitude=amplitude,
                    )

    else:
        length_params = _make_parallel_sweep_parameters(
            qubits,
            q_durations_list,
            "swap_duration",
        )
        with dsl.sweep(
            uid="swap_cal_len_nt",
            parameter=length_params,
        ):
            with dsl.acquire_loop_rt(**rt_kwargs):
                for q, length_param, q_amps in zip(
                    qubits,
                    length_params,
                    q_amplitudes_list,
                    strict=False,
                ):
                    with dsl.sweep(
                        **_sweep_kwargs_1d(
                            uid=f"swap_cal_amp_{q.uid}",
                            param_name=f"swap_amplitude_{q.uid}",
                            values=q_amps,
                            opts=opts,
                        )
                    ) as amplitude:
                        _play_swap_sequence(
                            q=q,
                            qop=qop,
                            opts=opts,
                            length=length_param,
                            amplitude=amplitude,
                        )


def _play_swap_sequence(
    q: QuantumElement,
    qop: QuantumOperations,
    opts: SwapExperimentOptions,
    *,
    length: SweepParameter | None = None,
    amplitude: SweepParameter | None = None,
) -> None:
    """Play the common SWAP calibration pulse sequence for one qubit."""
    swap_kwargs = {"pulse": opts.swap_pulse}
    if length is not None:
        swap_kwargs["length"] = length
    if amplitude is not None:
        swap_kwargs["amplitude"] = amplitude

    qop.x180(q, n=0)
    qop.swap(q, **swap_kwargs)
    qop.measure(q, swap_handle(q.uid))
    qop.passive_reset(q)


def _make_parallel_sweep_parameters(
    qubits: list,
    values_list: list,
    base_name: str,
) -> list[SweepParameter]:
    """Create one SweepParameter per qubit for a parallel near-time sweep."""
    return [
        SweepParameter(f"{base_name}_{q.uid}", values)
        for q, values in zip(qubits, values_list, strict=False)
    ]


def _acquire_loop_rt_kwargs(opts: SwapExperimentOptions) -> dict:
    """Build kwargs for dsl.acquire_loop_rt."""
    return {
        "count": opts.count,
        "averaging_mode": opts.averaging_mode,
        "acquisition_type": opts.acquisition_type,
        "repetition_mode": opts.repetition_mode,
        "repetition_time": opts.repetition_time,
        "reset_oscillator_phase": opts.reset_oscillator_phase,
    }


def _sweep_kwargs_1d(
    uid: str,
    param_name: str,
    values: np.ndarray,
    opts: SwapExperimentOptions,
) -> dict:
    """Common sweep kwargs builder for 1D real-time sweeps."""
    sweep_kwargs = {
        "uid": uid,
        "parameter": SweepParameter(param_name, values),
    }
    if opts.chunk_count > 0:
        sweep_kwargs["chunk_count"] = opts.chunk_count
    else:
        sweep_kwargs["auto_chunking"] = True
    return sweep_kwargs


def _get_sweep_mode(
    swap_durations: QubitSweepPoints | None,
    swap_amplitudes: QubitSweepPoints | None,
) -> str:
    """Infer sweep mode from which inputs are provided."""
    if swap_durations is not None and swap_amplitudes is not None:
        return "2d"
    if swap_durations is not None:
        return "length"
    return "amplitude"


def swap_handle(qubit_uid: str) -> str:
    """Return the result handle for a SWAP calibration measurement."""
    return f"{qubit_uid}/swap_cal"
