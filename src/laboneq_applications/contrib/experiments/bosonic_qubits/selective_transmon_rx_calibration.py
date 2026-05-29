# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Selective Transmon Rx calibration workflow.

The selective transmon Rx experiment calibrates the X180 and
X90 pulse amplitudes for photon-number-selective transmon drives.

Due to the dispersive interaction, the transmon frequency depends
on the photon number n in the memory cavity:

    f_transmon(n) = f_transmon(0) + n * chi

Each photon number requires independently calibrated pulse
amplitudes. This experiment sweeps the selective drive amplitude
at each number-dependent frequency and fits the resulting Rabi
oscillation to extract the pi and pi/2 amplitudes.

Pulse sequence per (qubit, photon number n, amplitude)::

    qb --- [ D(n_bar=n) ] --- [ Selective Rx(amp, n) ] --- [ measure ]

where:
- D(n_bar=n) prepares the memory cavity in a coherent state with
  mean photon number n̄ ≈ n. The displacement operation converts
  n̄ to a hardware amplitude via |β| = √n̄ and the calibrated
  parameter ``displacement_amp_per_unit_beta``.
  For n=0, no displacement is applied (cavity in vacuum).
- Selective Rx sweeps the transmon drive amplitude at frequency
  f_transmon(n), using the ``drive_transmon_at_n{n}`` signal line.

The Rabi oscillation contrast for n > 0 is proportional to the
population P(n|β) of the prepared coherent state at photon number
n. This does not affect the oscillation period, so the extracted
pi amplitude is still accurate.

Prerequisites:
    - Transmon spectroscopy (to find f_transmon(0)).
    - Displacement calibration (to know amp → |β| mapping for
      preparing coherent states at each n, i.e.
      ``displacement_amp_per_unit_beta`` must be set).

If multiple qubits are passed to the ``experiment_workflow``,
the pulse sequences are applied in parallel on all qubits.
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

from laboneq_applications.contrib.analysis.bosonic_qubits import (
    selective_transmon_rx_calibration as selective_transmon_rx_calibration_analysis,
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

import numpy as np

# ═════════════════════════════════════════════════════════════
# Options
# ═════════════════════════════════════════════════════════════


@workflow.task_options(base_class=TuneupExperimentOptions)
class SelectiveTransmonRxExperimentOptions:
    """Options for the selective transmon Rx calibration experiment.

    Additional attributes:
        selective_drive_pulse:
            Dictionary of pulse overrides for the selective
            transmon drive pulse. If None, the qubit parameter
            ``selective_drive_pulse`` is used.
        selective_drive_length:
            Override for the selective drive pulse length in
            seconds. If None, the qubit parameter
            ``selective_drive_length`` is used.
        memory_drive_pulse:
            Dictionary of pulse overrides for the memory cavity
            displacement pulse used for state preparation.
            If None, the qubit parameter ``memory_drive_pulse``
            is used.
        memory_drive_length:
            Override for the memory drive pulse length in
            seconds. If None, the qubit parameter
            ``memory_drive_length`` is used.
    """

    selective_drive_pulse: dict | None = None
    selective_drive_length: float | None = None
    memory_drive_pulse: dict | None = None
    memory_drive_length: float | None = None


@workflow.workflow_options
class SelectiveTransmonRxWorkflowOptions:
    """Options for the selective transmon Rx calibration workflow.

    Attributes:
        do_analysis:
            Whether to run the analysis workflow after the
            experiment. Default: True.
    """

    do_analysis: bool = True


# ═════════════════════════════════════════════════════════════
# Workflow
# ═════════════════════════════════════════════════════════════


@workflow.workflow(name="selective_transmon_Rx_calibration")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: list[str] | str,
    *,
    amplitudes_transmon: QubitSweepPoints,
    photon_numbers_memory: list[int],
    temporary_parameters: dict[str, dict | QuantumParameters] | None = None,
    options: SelectiveTransmonRxWorkflowOptions | None = None,
) -> None:
    """The selective transmon Rx calibration workflow.

    The workflow consists of the following steps:

    - [create_experiment]()
    - [compile_experiment]()
    - [run_experiment]()
    - [analysis_workflow]()

    Steps that can be added in the future include:

    - [update_qpu]()

    For each photon number n in ``photon_numbers_memory``, the memory
    cavity is prepared in a coherent state with mean photon number
    n̄ = n using the calibrated displacement operation, and the
    selective transmon drive amplitude is swept. The resulting Rabi
    oscillation is fitted to extract the pi and pi/2 pulse
    amplitudes.

    Arguments:
        session:
            The connected session to use for running the
            experiment.
        qpu:
            The qpu consisting of the original qubits and
            quantum operations.
        qubits:
            The qubits to run the experiments on, passed by UID.
            May be either a single qubit or a list of qubits.
        amplitudes_transmon:
            The selective drive amplitudes to sweep over for each
            qubit. Either a single array (broadcast to all
            qubits) or a list of arrays (one per qubit).
        photon_numbers_memory:
            List of cavity photon numbers to calibrate.
            For example, ``[0, 1, 2, 3, 4, 5]``.
        temporary_parameters:
            Temporary parameters to update the qubits with.
        options:
            The options for building the workflow.

    Returns:
        result:
            The result of the workflow.

    Example:
        ```python
        options = experiment_workflow.options()
        options.count(1024)
        result = experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=qpu.quantum_element_uids,
            amplitudes_transmon=[np.linspace(0.0, 1.0, 51)],
            photon_numbers_memory=[0, 1, 2, 3],
            options=options,
        ).run()
        ```
    """
    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    qubits = temporary_quantum_elements_from_qpu(temp_qpu, qubits)
    exp = create_experiment(
        qpu,
        qubits,
        amplitudes_transmon=amplitudes_transmon,
        photon_numbers_memory=photon_numbers_memory,
    )
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)
    with workflow.if_(options.do_analysis):
        selective_transmon_rx_calibration_analysis.analysis_workflow(
            result,
            qpu,
            qubits,
            amplitudes_transmon=amplitudes_transmon,
            photon_numbers_memory=photon_numbers_memory,
        )
    workflow.return_(result)


# ═════════════════════════════════════════════════════════════
# Task
# ═════════════════════════════════════════════════════════════


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubits: QuantumElements,
    amplitudes_transmon: QubitSweepPoints,
    photon_numbers_memory: list[int],
    options: SelectiveTransmonRxExperimentOptions | None = None,
) -> Experiment:
    """Creates a selective transmon Rx calibration experiment.

    For each photon number n, prepares a coherent state with
    n̄ = n via the displacement operation, then sweeps the
    selective transmon drive amplitude to observe Rabi
    oscillations.

    Arguments:
        qpu:
            The qpu consisting of the original qubits and
            quantum operations.
        qubits:
            The qubits to run the experiment on.
        amplitudes_transmon:
            The selective drive amplitudes to sweep for each
            qubit.
        photon_numbers_memory:
            List of cavity photon numbers to calibrate.
        options:
            The experiment options.

    Returns:
        The LabOne Q experiment.

    Raises:
        ValueError:
            If ``displacement_amp_per_unit_beta`` is not set
            on the qubit parameters and n > 0 is requested.
    """
    opts = SelectiveTransmonRxExperimentOptions() if options is None else options
    qubits, amplitudes_transmon = validation.validate_and_convert_qubits_sweeps(
        qubits,
        amplitudes_transmon,
    )

    # Outer sweep: photon number — same list for every qubit, swept in lockstep.
    photon_number_params = [
        SweepParameter(f"n_{q.uid}", np.array(photon_numbers_memory, dtype=int))
        for q in qubits
    ]

    # Inner sweep: selective-drive amplitude, one SweepParameter per qubit.
    amplitude_params = [
        SweepParameter(f"rabi_amp_{q.uid}", q_amplitudes)
        for q, q_amplitudes in zip(qubits, amplitudes_transmon, strict=False)
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
            name="photon_number_sweep",
            parameter=photon_number_params,
        ):
            with dsl.sweep(
                name="amplitude_sweep",
                parameter=amplitude_params,
            ):
                # For each qubit, branch on its own photon-number sweep handle.
                for q, n_param, amp_param in zip(
                    qubits,
                    photon_number_params,
                    amplitude_params,
                    strict=False,
                ):
                    with dsl.match(sweep_parameter=n_param):
                        for n in photon_numbers_memory:
                            with dsl.case(int(n)):
                                # Prepare coherent state with n̄ = n
                                if n > 0:
                                    qop.displacement(
                                        q,
                                        n_bar=n,
                                        length=opts.memory_drive_length,
                                        pulse=opts.memory_drive_pulse,
                                    )
                                # Selective transmon drive on |g,n>->|e,n>
                                qop.rx(
                                    q,
                                    angle=None,
                                    n=n,
                                    amplitude=amp_param,
                                    length=opts.selective_drive_length,
                                    pulse=opts.selective_drive_pulse,
                                )
                                qop.measure(q, rabi_result_handle(q.uid, n))
                                qop.passive_reset(q)


# ═════════════════════════════════════════════════════════════
# Handles
# ═════════════════════════════════════════════════════════════


def rabi_result_handle(qubit_uid: str, n: int) -> str:
    """Return the result handle for a selective Rabi measurement.

    Arguments:
        qubit_uid: The qubit UID.
        n: The photon number.

    Returns:
        The result handle string.
    """
    return f"{qubit_uid}/rabi_n{n}"
