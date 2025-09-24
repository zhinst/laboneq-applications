# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the TWPA spectroscopy experiment.

In this experiment, we sweep the frequency of a measure pulse to characterize
the TWPA response.

The TWPA spectroscopy experiment has the following pulse sequence:

    TWPA --- [ measure ]

This experiment only supports 1 TWPA at the time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq import workflow
from laboneq.simple import Experiment, SweepParameter, dsl
from laboneq.workflow.tasks import (
    compile_experiment,
    run_experiment,
)

from laboneq_applications.core import validation
from laboneq_applications.experiments.options import (
    TuneUpWorkflowOptions,
    TWPASpectroscopyExperimentOptions,
)
from laboneq_applications.tasks import (
    temporary_qpu,
    temporary_quantum_elements_from_qpu,
)

if TYPE_CHECKING:
    from laboneq.dsl.quantum import QPU
    from laboneq.dsl.session import Session
    from numpy.typing import ArrayLike

    from laboneq_applications.qpu_types.twpa.twpa_types import (
        TWPA,
        TWPAParameters,
    )


@workflow.workflow(name="twpa_spectroscopy")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    parametric_amplifier: TWPA,
    frequencies: ArrayLike,
    temporary_parameters: dict[str, dict | TWPAParameters] | None = None,
    options: TuneUpWorkflowOptions | None = None,
) -> None:
    """The TWPA Spectroscopy Workflow.

    The workflow consists of the following steps:

    - [create_experiment]()
    - [compile_experiment]()
    - [run_experiment]()
    - [analysis_workflow]()
    - [update_qpu]()

    Arguments:
        session:
            The connected session to use for running the experiment.
        qpu:
            The qpu consisting of the original TWPA and quantum operations.
        parametric_amplifier:
            The parametric amplifier to run the experiments on.
        frequencies:
            The readout frequencies to sweep over for the measure pulse (or CW)
            sent to the TWPA. Must be a list of numbers or an array.
        temporary_parameters:
            The temporary parameters to update the TWPA with.
        options:
            The options for building the workflow.
            In addition to options from [WorkflowOptions]

    Returns:
        result:
            The result of the workflow.

    Example:
        ```python
        options = experiment_workflow.options()
        options.spectroscopy_reset_delay(3e-6)
        twpa = TWPA("twpa0")
        qpu = QPU(
            pas=[twpa],
            quantum_operations=TWPAOperations(),
        )
        temp_qubits = qpu.copy_qubits()
        result = experiment_workflow(
            session=session,
            qpu=qpu,
            parametric_amplifier=twpa,
            frequencies=np.linspace(7.1e9, 7.6e9, 501),
        ).run()
        ```
    """
    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    parametric_amplifier = temporary_quantum_elements_from_qpu(
        temp_qpu, parametric_amplifier
    )

    exp = create_experiment(
        qpu,
        parametric_amplifier,
        frequencies=frequencies,
    )
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)
    workflow.return_(result)


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    parametric_amplifier: TWPA,
    frequencies: ArrayLike,
    options: TWPASpectroscopyExperimentOptions | None = None,
) -> Experiment:
    """Creates a TWPA Spectroscopy experiment.

    Arguments:
        qpu:
            The qpu consisting of the original TWPA and quantum operations.
        parametric_amplifier:
            The parametric amplifier to run the experiments on.
        frequencies:
            The readout frequencies to sweep over for the measure pulse (or CW)
            sent to the TWPA. Must be a list of numbers or an array.
        options:
            The options for building the experiment.
            See [TWPASpectroscopyExperimentOptions] and [BaseExperimentOptions] for
            accepted options.
            Overwrites the options from [BaseExperimentOptions].

    Returns:
        experiment:
            The generated LabOne Q experiment instance to be compiled and executed.

    Example:
        ```python
        options = {
            "count": 10,
            "spectroscopy_reset_delay": 3e-6
        }
        options = TuneupExperimentOptions(**options)
        setup = DeviceSetup()
        qpu = QPU(
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            quantum_operations=TunableTransmonOperations(),
        )
        create_experiment(
            qpu=qpu,
            parametric_amplifier=twpa,
            frequencies=np.linspace(7.1e9, 7.6e9, 501),
            options=options,
        )
        ```
    """
    # Define the custom options for the experiment
    opts = TWPASpectroscopyExperimentOptions() if options is None else options
    frequencies = validation.validate_and_convert_sweeps_to_arrays(frequencies)

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
            name=f"freq_{parametric_amplifier.uid}",
            parameter=SweepParameter(
                f"frequencies_{parametric_amplifier.uid}", frequencies
            ),
        ) as frequency:
            qop.set_readout_frequency(parametric_amplifier, frequency)
            qop.twpa_acquire(
                parametric_amplifier,
                dsl.handles.result_handle(parametric_amplifier.uid),
            )
            qop.twpa_delay(parametric_amplifier, opts.spectroscopy_reset_delay)
