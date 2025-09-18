# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the cancellation tone calibration workflow.

In this experiment, we measure the amplitude and phase of the pump tone while sweeping
the phase shift and attenuation of the cancellation tone.

The cancellation tone calibration experiment has the following pulse sequence:

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

from laboneq_applications.contrib.analysis.calibrate_cancellation import (
    analysis_workflow,
)
from laboneq_applications.core import validation
from laboneq_applications.experiments.options import (
    TuneUpWorkflowOptions,
    TWPATuneUpExperimentOptions,
)
from laboneq_applications.tasks import (
    temporary_qpu,
    temporary_quantum_elements_from_qpu,
    update_qubits,
)

if TYPE_CHECKING:
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.session import Session
    from numpy.typing import ArrayLike

    from laboneq_applications.qpu_types.twpa.twpa_types import (
        TWPA,
        TWPAParameters,
    )


@workflow.workflow(name="calibrate_cancellation")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    parametric_amplifier: TWPA,
    cancel_phase: ArrayLike,
    cancel_attenuation: ArrayLike,
    temporary_parameters: dict[str, dict | TWPAParameters] | None = None,
    options: TuneUpWorkflowOptions | None = None,
) -> None:
    """The cancellation tone calibration Workflow.

    The workflow consists of the following steps:

    - [create_experiment]()
    - [compile_experiment]()
    - [run_experiment]()
    - [analysis_workflow]()
    - [update_qubits]()

    Arguments:
        session:
            The connected session to use for running the experiment.
        qpu:
            The qpu consisting of the original pas and quantum operations.
        parametric_amplifier:
            The parametric amplifier to run the experiments on.
        cancel_phase:
            The phase shift of the cancellation tone.
            Must be a list of numbers or an array.
        cancel_attenuation:
            The attenuation of the cancellation tone.
            Must be a list of numbers or an array.
        temporary_parameters:
            The temporary parameters to update the parametric amplifiers with.
        options:
            The options for building the workflow.

    Returns:
        result:
            The result of the workflow.

    Example:
        ```python
        options = experiment_workflow.options()
        options.create_experiment.count(10)
        pa = TWPA("twpa0")
        qpu = QPU(
            qubits=[pa],
            quantum_operations=TWPAOperations(),
        )
        temp_pas = qpu_copy_elements()
        result = experiment_workflow(
            session=session,
            qpu=qpu,
            parametric_amplifier=pa,
            cancel_phase=[0.0, 0.1],
            cancel_attenuation=[0.1, 0.5],
            options=options,
        ).run()
        ```
    """
    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    parametric_amplifier = temporary_quantum_elements_from_qpu(
        temp_qpu, parametric_amplifier
    )
    exp_on = create_experiment(
        qpu,
        parametric_amplifier,
        cancel_phase=cancel_phase,
        cancel_attenuation=cancel_attenuation,
        cancellation_on=True,
    )
    compiled_exp_on = compile_experiment(session, exp_on)
    result_on = run_experiment(session, compiled_exp_on)

    exp_off = create_experiment(
        qpu,
        parametric_amplifier,
        cancel_phase=cancel_phase,
        cancel_attenuation=cancel_attenuation,
        cancellation_on=False,
    )
    compiled_exp_off = compile_experiment(session, exp_off)
    result_off = run_experiment(session, compiled_exp_off)

    with workflow.if_(options.do_analysis):
        analysis_results = analysis_workflow(
            result_on,
            result_off,
            parametric_amplifier,
            cancel_phase,
            cancel_attenuation,
        )
        parametric_amplifier_parameters = analysis_results.output
        with workflow.if_(options.update):
            update_qubits(qpu, parametric_amplifier_parameters["new_parameter_values"])
    workflow.return_(data=result_on, ref=result_off)


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    parametric_amplifier: TWPA,
    cancel_phase: ArrayLike,
    cancel_attenuation: ArrayLike,
    cancellation_on: bool = False,  # noqa: FBT001, FBT002
    options: TWPATuneUpExperimentOptions | None = None,
) -> Experiment:
    """Creates a cancellation tone calibration Experiment.

    Arguments:
        qpu:
            The qpu consisting of the original pas and quantum operations.
        parametric_amplifier:
            The parametric amplifier to run the experiments on.
        cancel_phase:
            The phase shift of the cancellation tone.
            Must be a list of numbers or an array.
        cancel_attenuation:
            The attenuation of the cancellation tone.
            Must be a list of numbers or an array.
        cancellation_on:
            Enable or disable the cancellation tone.
        options:
            The options for building the experiment.
            See [TWPATuneUpExperimentOptions] and [BaseExperimentOptions] for
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
        options = TWPATuneUpExperimentOptions(**options)
        setup = DeviceSetup()
        pa = TWPA("twpa0")
        qpu = QPU(
            qubits=[pa],
            quantum_operations=TunableTransmonOperations(),
        )
        create_experiment(
            qpu=qpu,
            parametric_amplifier=pa,
            frequencies=np.linspace(7.1e9, 7.6e9, 501),
            options=options,
        )
        ```
    """
    opts = TWPATuneUpExperimentOptions() if options is None else options
    cancel_phase = validation.validate_and_convert_sweeps_to_arrays(cancel_phase)
    cancel_attenuation = validation.validate_and_convert_sweeps_to_arrays(
        cancel_attenuation
    )

    qop = qpu.quantum_operations
    phase = SweepParameter(f"phases_{parametric_amplifier.uid}", cancel_phase)
    attenuation = SweepParameter(
        f"attenuation_{parametric_amplifier.uid}", cancel_attenuation
    )
    # The probe tone is generated from the pump tone of the SHFPPC.
    qop.set_readout_amplitude.omit_section(parametric_amplifier, 0)
    if cancellation_on:
        with dsl.acquire_loop_rt(
            count=opts.count,
            averaging_mode=opts.averaging_mode,
            acquisition_type=opts.acquisition_type,
            repetition_mode=opts.repetition_mode,
            repetition_time=opts.repetition_time,
            reset_oscillator_phase=opts.reset_oscillator_phase,
        ):
            with dsl.sweep(
                name=f"attenuations_{parametric_amplifier.uid}",
                parameter=SweepParameter(
                    f"attenuation_{parametric_amplifier.uid}", cancel_attenuation
                ),
            ) as attenuation:
                with dsl.sweep(
                    name=f"phase_{parametric_amplifier.uid}",
                    parameter=SweepParameter(
                        f"phases_{parametric_amplifier.uid}", cancel_phase
                    ),
                ) as phase:
                    qop.set_pump_cancellation(
                        parametric_amplifier, attenuation, phase, True
                    )
                    qop.twpa_acquire(
                        parametric_amplifier,
                        dsl.handles.result_handle(parametric_amplifier.uid),
                    )
                    qop.twpa_delay(parametric_amplifier, opts.spectroscopy_reset_delay)
    else:
        with dsl.acquire_loop_rt(
            count=opts.count,
            averaging_mode=opts.averaging_mode,
            acquisition_type=opts.acquisition_type,
            repetition_mode=opts.repetition_mode,
            repetition_time=opts.repetition_time,
            reset_oscillator_phase=opts.reset_oscillator_phase,
        ):
            qop.set_pump_cancellation(parametric_amplifier, attenuation, phase, False)
            qop.twpa_acquire(
                parametric_amplifier,
                dsl.handles.result_handle(parametric_amplifier.uid),
            )
            qop.twpa_delay(parametric_amplifier, opts.spectroscopy_reset_delay)
