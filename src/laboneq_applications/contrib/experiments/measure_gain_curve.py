# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the gain curve measurement workflow.

In this experiment, we measure the probe tone spectrum as a function of the pump power
and probe frequency.

The gain curve experiment has the following pulse sequence:

    TWPA --- [ measure ]

This experiment only supports 1 TWPA at the time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq import workflow
from laboneq.dsl.enums import AcquisitionType
from laboneq.simple import Experiment, SweepParameter, dsl
from laboneq.workflow.tasks import (
    compile_experiment,
    run_experiment,
)

from laboneq_applications.contrib.analysis.measure_gain_curve import analysis_workflow
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
    from laboneq_applications.qpu_types.twpa import (
        TWPA,
        TWPAParameters,
    )

if TYPE_CHECKING:
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.session import Session
    from numpy.typing import ArrayLike


@workflow.workflow(name="measure_gain_curve")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    parametric_amplifier: TWPA,
    probe_frequency: ArrayLike,
    pump_power: ArrayLike,
    selected_indexes: list | None = None,
    temporary_parameters: dict[str, dict | TWPAParameters] | None = None,
    options: TuneUpWorkflowOptions | None = None,
) -> None:
    """The gain curve measurement Workflow.

    The workflow consists of the following steps:

    - [create_experiment]()
    - [compile_experiment]()
    - [run_experiment]()
    - [analysis_workflow]()
    - [update_pas]()

    Arguments:
        session:
            The connected session to use for running the experiment.
        qpu:
            The qpu consisting of the original parametric amplifiers
            and quantum operations.
        parametric_amplifier:
            The parametric amplifier to run the experiments on.
        probe_frequency:
            The pump frequencies to sweep over sent to the parametric amplifier.
            Must be a list of numbers or an array.
        pump_power:
            The pump powers to sweep over sent to the pa.
            Must be a list of numbers or an array.
        selected_indexes:
            Select the indexes of the pump_power for plotting.
        temporary_parameters:
            The temporary parameters to update the parametric amplifiers with.
        options:
            The options for building the workflow.
            In addition to options from [WorkflowOptions], the following
            custom options are supported:
                - create_experiment: The options for creating the experiment.

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
            quantum_operations=paOperations(),
        )
        result = experiment_workflow(
            session=session,
            qpu=qpu,
            parametric_amplifier=pa,
            pump_power=np.linspace(0, 10, 501),
            probe_frequency=np.linspace(7.1e9, 7.6e9, 501),
            options=options,
        ).run()
        ```
    """
    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    parametric_amplifier = temporary_quantum_elements_from_qpu(
        temp_qpu, parametric_amplifier
    )
    exp1 = create_experiment(
        qpu,
        parametric_amplifier=parametric_amplifier,
        probe_frequency=probe_frequency,
        pump_power=pump_power,
        pump_on=True,
    )
    compiled_exp1 = compile_experiment(session, exp1)
    result_pump_on = run_experiment(session, compiled_exp1)

    exp2 = create_experiment(
        qpu,
        parametric_amplifier=parametric_amplifier,
        probe_frequency=probe_frequency,
        pump_power=pump_power,
        pump_on=False,
    )
    compiled_exp2 = compile_experiment(session, exp2)
    result_pump_off = run_experiment(session, compiled_exp2)
    with workflow.if_(options.do_analysis):
        _ = analysis_workflow(
            result_pump_on,
            result_pump_off,
            parametric_amplifier,
            probe_frequency,
            pump_power,
            selected_indexes,
        )
    workflow.return_(result_pump_on=result_pump_on, result_pump_off=result_pump_off)


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    parametric_amplifier: TWPAParameters,
    probe_frequency: ArrayLike,
    pump_power: ArrayLike,
    pump_on: bool = False,  # noqa: FBT001, FBT002
    options: TWPASpectroscopyExperimentOptions | None = None,
) -> Experiment:
    """Creates a Phase Diagram Experiment.

    Arguments:
        qpu:
            The qpu consisting of the original pas and quantum operations.
        parametric_amplifier:
            The parametric amplifier to run the experiments on.
        probe_frequency:
            The pump frequencies to sweep over sent to the parametric amplifiers.
            Must be a list of numbers or an array.
        pump_power:
            The pump powers to sweep over sent to the parametric amplifiers.
            Must be a list of numbers or an array.
        pump_on:
            The flag to turn on the pump.
        options:
            The options for building the experiment.
            See [TWPASpectroscopyExperimentOptions] and [BaseExperimentOptions] for
            accepted options.
            Overwrites the options from [TWPASpectroscopyExperimentOptions].
            If the `use_probe_from_ppc` option is set to True, the acquisition type
            is set to AcquisitionType.SPECTROSCOPY_PSD.
            Otherwise, the acquisition type is set to AcquisitionType.SPECTROSCOPY.

    Returns:
        experiment:
            The generated LabOne Q experiment instance to be compiled and executed.

    Example:
        ```python
        options = TWPASpectroscopyExperimentOptions()
        setup = DeviceSetup()
        pa = TWPA("twpa0")
        qpu = QPU(
            qubits=[pa],
            quantum_operations=paOperations(),
        )
        create_experiment(
            qpu=qpu,
            parametric_amplifier=pa,
            pump_power=np.linspace(0, 10, 501),
            probe_frequency=np.linspace(7.1e9, 7.6e9, 501),
            options=options,
        )
        ```
    """
    # Define the custom options for the experiment
    opts = TWPASpectroscopyExperimentOptions() if options is None else options
    probe_frequency = validation.validate_and_convert_sweeps_to_arrays(probe_frequency)
    pump_power = validation.validate_and_convert_sweeps_to_arrays(pump_power)

    if opts.use_probe_from_ppc:
        opts.acquisition_type = AcquisitionType.SPECTROSCOPY_PSD
    else:
        opts.acquisition_type = AcquisitionType.SPECTROSCOPY

    qop = qpu.quantum_operations
    frequency = SweepParameter(
        f"frequencies_{parametric_amplifier.uid}", probe_frequency
    )
    power = SweepParameter(f"powers_{parametric_amplifier.uid}", pump_power)

    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        with dsl.sweep(
            name=f"power_{parametric_amplifier.uid}",
            parameter=SweepParameter(f"powers_{parametric_amplifier.uid}", pump_power),
        ) as power:
            with dsl.sweep(
                name=f"freq_{parametric_amplifier.uid}",
                parameter=SweepParameter(
                    f"frequencies_{parametric_amplifier.uid}", probe_frequency
                ),
            ) as frequency:
                qop.set_readout_frequency(parametric_amplifier, frequency)
                qop.set_pump_power(parametric_amplifier, power)
                qop.twpa_acquire(
                    parametric_amplifier,
                    dsl.handles.result_handle(parametric_amplifier.uid),
                )
                qop.twpa_delay(parametric_amplifier, opts.spectroscopy_reset_delay)

                if opts.use_probe_from_ppc:
                    qop.set_readout_amplitude(parametric_amplifier, 0)

                # TODO: Replace this by an operation to set the amplifier
                # pump properties
                calibration = dsl.experiment_calibration()
                amplifier_pump = calibration[
                    parametric_amplifier.signals["acquire"]
                ].amplifier_pump
                amplifier_pump.pump_on = pump_on
                amplifier_pump.probe_on = opts.use_probe_from_ppc
                amplifier_pump.probe_frequency = (
                    frequency
                    if opts.use_probe_from_ppc
                    else parametric_amplifier.parameters.probe_frequency
                )
