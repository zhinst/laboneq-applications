# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the memory spectroscopy experiment for bosonic qubits.

The memory spectroscopy experiment has the following pulse sequence:

    qb --- [ memory spectroscopy drive (swept frequency) ] ---
    --- [ transmon x180 (when memory in |0⟩) ] --- [ measure ]

The idea is to probe the memory cavity resonance by sweeping the frequency
of a long spectroscopy tone applied to the memory cavity. After the
spectroscopy pulse, a transmon pi pulse (calibrated for the memory in |0⟩)
is applied. If the spectroscopy tone populated the memory, the transmon
frequency is shifted by chi and the pi pulse will not fully excite the
transmon, resulting in a change in the readout signal.

If multiple qubits are passed to the `run` workflow, the above pulses are
applied in parallel on all the qubits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq import workflow
from laboneq.dsl.quantum import QuantumParameters
from laboneq.simple import Experiment, SweepParameter, dsl
from laboneq.workflow.tasks import (
    compile_experiment,
    run_experiment,
)

from laboneq_applications.core import validation
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

    from laboneq_applications.typing import QuantumElements, QubitSweepPoints


@workflow.task_options(base_class=BaseExperimentOptions)
class MemorySpectroscopyExperimentOptions:
    """Options for the memory spectroscopy experiment.

    Additional attributes:
        spectroscopy_pulse:
            Dictionary of pulse overrides for the memory spectroscopy
            pulse. By default, the pulse parameters from the qubit's
            `memory_spectroscopy_pulse` are used.
    """

    spectroscopy_pulse: dict | None = None


@workflow.workflow_options
class MemorySpectroscopyWorkflowOptions:
    """Options for the memory spectroscopy workflow."""


@workflow.workflow(name="memory_spectroscopy")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: list[str] | str,
    *,
    frequencies: QubitSweepPoints,
    temporary_parameters: dict[str, dict | QuantumParameters] | None = None,
    options: MemorySpectroscopyWorkflowOptions | None = None,
) -> None:
    """The memory spectroscopy workflow.

    The workflow consists of the following steps:

    - [create_experiment]()
    - [compile_experiment]()
    - [run_experiment]()

    Steps that can be added in the future include:

    - [analysis_workflow]()
    - [update_qpu]()

    Arguments:
        session:
            The connected session to use for running the experiment.
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubits to run the experiments on, passed by UID. May be
            either a single qubit or a list of qubits.
        frequencies:
            The RF frequencies to sweep over for the memory cavity
            spectroscopy drive for each qubit. Note that `frequencies`
            must be identical for qubits that use the same measure port.
        temporary_parameters:
            The temporary parameters to update the qubits with.
        options:
            The options for building the workflow.

    Returns:
        result:
            The result of the workflow.

    Example:
        ```python
        options = experiment_workflow.options()
        options.count(10)
        qpu = QPU(
            qubits=[BosonicQubit("q0"), BosonicQubit("q1")],
            quantum_operations=BosonicQubitOperations(),
        )
        result = experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=["q0", "q1"],
            frequencies=[
                np.linspace(6.0e9, 6.2e9, 101),
                np.linspace(6.0e9, 6.2e9, 101),
            ],
            options=options,
        ).run()
        ```
    """
    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    qubits = temporary_quantum_elements_from_qpu(temp_qpu, qubits)
    exp = create_experiment(
        qpu,
        qubits,
        frequencies=frequencies,
    )
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)
    workflow.return_(result)


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubits: QuantumElements,
    frequencies: QubitSweepPoints,
    options: MemorySpectroscopyExperimentOptions | None = None,
) -> Experiment:
    """Creates a memory spectroscopy experiment.

    The pulse sequence for each qubit is:

    1. Memory spectroscopy drive with swept frequency.
    2. Transmon pi pulse (x180), calibrated for the memory cavity in |0⟩.
    3. Readout pulse and acquisition.

    When the spectroscopy tone is resonant with the memory cavity, it
    populates the cavity. This shifts the transmon frequency by the
    dispersive shift chi, causing the subsequent pi pulse to be
    off-resonant and resulting in a measurable change in the readout
    signal.

    Arguments:
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubits to run the experiments on. May be either a single
            qubit or a list of qubits.
        frequencies:
            The RF frequencies to sweep over for the memory cavity
            spectroscopy drive for each qubit. Note that `frequencies`
            must be identical for qubits that use the same measure port.
        options:
            The options for building the experiment.
            See [MemorySpectroscopyExperimentOptions] and
            [BaseExperimentOptions] for accepted options.

    Returns:
        experiment:
            The generated LabOne Q experiment instance to be compiled
            and executed.

    Raises:
        ValueError:
            If frequencies is not a list of numbers or array when a
            single qubit is passed.

    Example:
        ```python
        options = {
            "count": 10,
            "averaging_mode": "cyclic",
            "acquisition_type": "integration_trigger",
            "cal_traces": True,
        }
        options = MemorySpectroscopyExperimentOptions(**options)
        qpu = QPU(
            qubits=[BosonicQubit("q0"), BosonicQubit("q1")],
            quantum_operations=BosonicQubitOperations(),
        )
        create_experiment(
            qpu=qpu,
            qubits=["q0", "q1"],
            frequencies=[
                np.linspace(6.0e9, 6.2e9, 101),
                np.linspace(6.0e9, 6.2e9, 101),
            ],
            options=options,
        )
        ```
    """
    opts = MemorySpectroscopyExperimentOptions() if options is None else options
    qubits, frequencies = validation.validate_and_convert_qubits_sweeps(
        qubits, frequencies
    )

    # Build one SweepParameter per qubit, all driven by the same outer sweep.
    sweep_params = [
        SweepParameter(f"frequency_{q.uid}", q_frequencies)
        for q, q_frequencies in zip(qubits, frequencies, strict=False)
    ]

    qop = qpu.quantum_operations
    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        with dsl.sweep(
            name="frequency_sweep",
            parameter=sweep_params,
        ):
            for q, frequency in zip(qubits, sweep_params, strict=False):
                qop.set_frequency(q, frequency, target="memory")
                qop.memory_spectroscopy_drive(
                    q,
                    pulse=opts.spectroscopy_pulse,
                )
                qop.x180(q, n=0)
                qop.measure(q, dsl.handles.result_handle(q.uid))
                qop.passive_reset(q)
