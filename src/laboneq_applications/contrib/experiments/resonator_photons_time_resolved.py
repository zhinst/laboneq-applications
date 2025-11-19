# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the resonator photon number time resolution experiment.

In this experiment, we perform a short qubit spectroscopy during the qubit readout,
and sweeping the spectroscopy pulse over the measurement pulse to time resolve the
photon number n. For superconducting qubits the qubit frequency changes with resonator
photons according to:

    Δf_ge(t) = -2𝛘n(t)

The pulse sequence for the resonator photon number time resolution experiment is as
follows:

   ----------------------- [ delay(sweep)][ spectr ]
qb - [ prep transition ] - [    measure inspect    ] - [ clear time ] - [ measure ]

If multiple qubits are passed to the `run` workflow, the above pulses are applied
in parallel on all the qubits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from laboneq.simple import SectionAlignment, SweepParameter, dsl
from laboneq.workflow import if_, option_field, return_, task, task_options, workflow
from laboneq.workflow.tasks import (
    compile_experiment,
    run_experiment,
)

from laboneq_applications.contrib.analysis import resonator_photons_time_resolved
from laboneq_applications.core.validation import validate_and_convert_qubits_sweeps
from laboneq_applications.experiments.options import (
    QubitSpectroscopyExperimentOptions,
    TuneUpWorkflowOptions,
)
from laboneq_applications.tasks.parameter_updating import temporary_modify

if TYPE_CHECKING:
    from laboneq.dsl.quantum import TransmonParameters
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.session import Session

    from laboneq_applications.typing import QuantumElements, Qubits, QubitSweepPoints


# create additional options for the QNDness experiment
@task_options(base_class=QubitSpectroscopyExperimentOptions)
class ResonatorPhotonsExperimentOptions:
    """Options for the time-resolved resonator photons experiment.

    Additional attributes:
        delay_between_measurements:
            Time between inspect measure pulse and spectroscopy measure for better
            contrast.
    """

    delay_between_measurements: float = option_field(
        1e-6, description="Time delay between successive measurement operations."
    )


@workflow(name="resonator_photons_time_resolved")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: Qubits,
    times: QubitSweepPoints,
    frequencies: QubitSweepPoints,
    temporary_parameters: dict[str, dict | TransmonParameters] | None = None,
    options: TuneUpWorkflowOptions | None = None,
) -> None:
    """The Resonator Photon Number Time Resolution Workflow.

    The workflow consists of the following steps:

    - [create_experiment]()
    - [compile_experiment]()
    - [run_experiment]()
    - [analysis_workflow]()

    Arguments:
        session:
            The connected session to use for running the experiment.
        qpu:
            The QPU consisting of the original qubits and quantum operations.
        qubits:
            The qubits to run the experiments on. May be either a single
            qubit or a list of qubits. Multi qubit operation is only possible
            with multiple QA channels.
        times:
            The array of times that were swept over the measurement pulse in the
            experiment. If `qubits` is a single qubit, `times` must be a list of
            numbers or an array.Otherwise, it must be a list of lists of numbers
            or arrays.
        frequencies:
            The array of frequencies that were swept over in the experiment. If `qubits`
            is a single qubit, `frequencies` must be a list of numbers or an array.
            Otherwise, it must be a list of lists of numbers or arrays.
        temporary_parameters:
            The temporary parameters to update the qubits with.
        options:
            The options for building the workflow.
            In addition to options from [WorkflowOptions], the following
            custom options are supported:
                - `create_experiment`: The options for creating the experiment.

    Returns:
        WorkflowBuilder:
            The builder for the experiment workflow.

    Example:
        ```python
        options = resonator_photons_time_resolved.experiment_workflow.options()
        options.count(2**9)
        options.delay_between_measurements(1e-6)
        qpu = QPU(
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            quantum_operations=TunableTransmonOperations(),
        )

        temporary_parameters = {}
        for q in qubits:
            temp_pars = deepcopy(q.parameters)
            temp_pars.drive_range = -10
            temp_pars.spectroscopy_amplitude = 0.25
            temp_pars.spectroscopy_length = 80e-9
            temporary_parameters[q.uid] = temp_pars

        result = resonator_photons_time_resolved.experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=qpu.quantum_elements[0],
            times=[
                np.linspace(0, 3e-6, 21),
                np.linspace(0, 3e-6, 21),
            ],
            frequencies=[
                np.linspace(center_q1 - off, center_q1 + off, 151),
                np.linspace(center_q2 - off, center_q2 + off, 151),
            ],
            temporary_parameters=temporary_parameters,
            options=options,
        ).run()
        ```
    """
    qubits = temporary_modify(qubits, temporary_parameters)
    exp = create_experiment(
        qpu,
        qubits,
        times,
        frequencies,
    )
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)
    with if_(options.do_analysis):
        resonator_photons_time_resolved.analysis_workflow(
            result, qubits, times, frequencies
        )
    return_(result)


@task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubits: QuantumElements,
    times: QubitSweepPoints,
    frequencies: QubitSweepPoints,
    options: ResonatorPhotonsExperimentOptions | None = None,
) -> None:
    """Creates a Photon Number Resolution Experiment.

    Arguments:
        qpu:
            The QPU consisting of the original qubits and quantum operations.
        qubits:
            The qubits to run the experiments on. May be either a single
            qubit or a list of qubits. Multi qubit operation is only possible
            with multiple QA channels.
        times:
            The array of times that were swept over the measurement pulse in the
            experiment. If `qubits` is a single qubit, `times` must be a list of numbers
            or an array. Otherwise, it must be a list of lists of numbers or arrays.
        frequencies:
            The array of frequencies that were swept over in the experiment. If `qubits`
            is a single qubit, `frequencies` must be a list of numbers or an array.
            Otherwise, it must be a list of lists of numbers or arrays.
        options:
            The options for building the experiment.
            See [ResonatorPhotonsExperimentOptions] and
            [QubitSpectroscopyExperimentOptions] for accepted options.

    Returns:
        experiment:
            The generated LabOne Q experiment instance to be compiled and executed.

    Raises:
        ValueError:
            If the `qubits`, `times`, and `frequencies` are not of the same length.

        ValueError:
            If `times` and `frequencies` are not a list of numbers when a single
            qubit is passed.

        ValueError:
            If `frequencies` is not a list of lists of numbers.
            If `times` is not a list of lists of numbers.

    Example:
        ```python
        options = resonator_photons_time_resolved.experiment_workflow.options()
        options.count(2**9)
        options.delay_between_measurements(1e-6)
        qpu = QPU(
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            quantum_operations=TunableTransmonOperations(),
        )
        temporary_parameters = {}
        for q in qubits:
            temp_pars = deepcopy(q.parameters)
            temp_pars.drive_range = -10
            temp_pars.spectroscopy_amplitude = 0.25
            temp_pars.spectroscopy_length = 80e-9
            temporary_parameters[q.uid] = temp_pars

        create_experiment(
            qpu=qpu,
            qubits=qpu.quantum_elements[0],
            times=[
                np.linspace(0, 3e-6, 21),
                np.linspace(0, 3e-6, 21),
            ],
            frequencies=[
                np.linspace(center_q1 - off, center_q1 + off, 151),
                np.linspace(center_q2 - off, center_q2 + off, 151),
            ],
            temporary_parameters=temporary_parameters,
            options=options,
        ).run()
        ```
    """
    # Define the custom options for the experiment
    opts = ResonatorPhotonsExperimentOptions() if options is None else options
    qop = qpu.quantum_operations

    qubits_validated, frequencies = validate_and_convert_qubits_sweeps(
        qubits, frequencies
    )
    qubits_validated, times = validate_and_convert_qubits_sweeps(qubits, times)
    max_measure_section_length = qpu.measure_section_length(qubits)
    max_inspect_section_length = np.max(times) + max(
        q.parameters.spectroscopy_length for q in qubits
    )

    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        for q, q_frequencies, q_times in zip(qubits_validated, frequencies, times):
            with dsl.sweep(
                name=f"time_sweep_{q.uid}",
                parameter=SweepParameter(f"sweep_{q.uid}", q_times),
            ) as time:
                with dsl.sweep(
                    name=f"freqs_{q.uid}",
                    parameter=SweepParameter(f"frequency_{q.uid}", q_frequencies),
                ) as frequency:
                    qop.set_frequency(q, frequency)
                    with dsl.section(
                        name="inspect",
                        alignment=SectionAlignment.LEFT,
                        length=max_inspect_section_length,
                    ):
                        # omit_reserves is required to play the spectroscopy
                        # pulse during the first readout.
                        sec = qop.measure.omit_reserves(q, f"inspect_{q.uid}")
                        qop.delay.omit_reserves(q, time)
                        qop.qubit_spectroscopy_drive.omit_reserves(q)
                    with dsl.section(name="delay", alignment=SectionAlignment.LEFT):
                        qop.delay(q, opts.delay_between_measurements)
                    with dsl.section(name="measure", alignment=SectionAlignment.LEFT):
                        sec = qop.measure(q, dsl.handles.result_handle(q.uid))
                        # Fix the length of the measure section
                        sec.length = max_measure_section_length
                        qop.passive_reset(q, delay=opts.spectroscopy_reset_delay)
