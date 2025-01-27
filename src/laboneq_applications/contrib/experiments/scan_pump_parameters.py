# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the TWPA tune up workflow.

In this experiment, we sweep the pump frequency and power at a fixed probe frequency
to characterize the parametric amplifier.

The TWPA tune up experiment has the following pulse sequence:

    TWPA --- [ measure ]

This experiment only supports 1 TWPA at the time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq import workflow
from laboneq.simple import AcquisitionType, Experiment, SweepParameter, dsl
from laboneq.workflow.tasks import (
    compile_experiment,
    run_experiment,
)

from laboneq_applications.contrib.analysis.scan_pump_parameters import analysis_workflow
from laboneq_applications.core import validation
from laboneq_applications.experiments.options import (
    TWPATuneUpExperimentOptions,
    TWPATuneUpWorkflowOptions,
)
from laboneq_applications.tasks.parameter_updating import (
    temporary_modify,
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


@workflow.workflow(name="scan_pump_parameters")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    parametric_amplifier: TWPA,
    pump_frequency: ArrayLike,
    pump_power: ArrayLike,
    temporary_parameters: dict[str, dict | TWPAParameters] | None = None,
    options: TWPATuneUpWorkflowOptions | None = None,
) -> None:
    """The Workflow for scanning the pump parameters of a TWPA.

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
            The qpu consisting of the original parametric amplifiers
            and quantum operations.
        parametric_amplifier:
            The parametric amplifier to run the experiments on.
        pump_frequency:
            The pump frequencies to sweep over sent to the parametric amplifier.
            Must be a list of numbers or an array.
        pump_power:
            The pump powers to sweep over sent to the parametric amplifier.
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
        twpa = TWPA("twpa0")
        qpu = QPU(
            qubits=[twpa],
            quantum_operations=TWPAOperations(),
        )
        result = experiment_workflow(
            session=session,
            qpu=qpu,
            parametric_amplifier=twpa,
            pump_power=np.linspace(0, 10, 501),
            pump_frequency=np.linspace(7.1e9, 7.6e9, 501),
            options=options,
        ).run()
        ```
    """
    parametric_amplifier = temporary_modify(parametric_amplifier, temporary_parameters)
    exp1 = create_experiment(
        qpu=qpu,
        parametric_amplifier=parametric_amplifier,
        pump_frequency=pump_frequency,
        pump_power=pump_power,
        pump_on=True,
        probe_on=True,
    )
    compiled_exp1 = compile_experiment(session, exp1)
    data_signal_pump_on = run_experiment(session, compiled_exp1)

    exp2 = create_experiment(
        qpu=qpu,
        parametric_amplifier=parametric_amplifier,
        pump_frequency=pump_frequency,
        pump_power=pump_power,
        pump_on=False,
        probe_on=True,
    )
    compiled_exp2 = compile_experiment(session, exp2)
    data_signal_pump_off = run_experiment(session, compiled_exp2)

    with workflow.if_(options.do_snr):
        exp3 = create_experiment(
            qpu=qpu,
            parametric_amplifier=parametric_amplifier,
            pump_frequency=pump_frequency,
            pump_power=pump_power,
            pump_on=True,
            probe_on=False,
        )
        compiled_exp3 = compile_experiment(session, exp3)
        data_noise_pump_on = run_experiment(session, compiled_exp3)

        exp4 = create_experiment(
            qpu=qpu,
            parametric_amplifier=parametric_amplifier,
            pump_frequency=[],
            pump_power=[],
            pump_on=False,
            probe_on=False,
        )
        compiled_exp4 = compile_experiment(session, exp4)
        data_noise_pump_off = run_experiment(session, compiled_exp4)

        with workflow.if_(options.do_analysis):
            analysis_results = analysis_workflow(
                data_signal_pump_on,
                data_signal_pump_off,
                parametric_amplifier,
                pump_frequency,
                pump_power,
                data_noise_pump_on,
                data_noise_pump_off,
            )
            parametric_amplifier_parameters = analysis_results.output
            with workflow.if_(options.update):
                update_qubits(
                    qpu, parametric_amplifier_parameters["new_parameter_values"]
                )

        workflow.return_(
            data_signal_pump_on=data_signal_pump_on,
            data_signal_pump_off=data_signal_pump_off,
            data_noise_pump_on=data_noise_pump_on,
            data_noise_pump_off=data_noise_pump_off,
        )

    with workflow.else_():
        with workflow.if_(options.do_analysis):
            analysis_results = analysis_workflow(
                data_signal_pump_on,
                data_signal_pump_off,
                parametric_amplifier,
                pump_frequency,
                pump_power,
            )
            parametric_amplifier_parameters = analysis_results.output
            with workflow.if_(options.update):
                update_qubits(
                    qpu, parametric_amplifier_parameters["new_parameter_values"]
                )
        workflow.return_(
            data_signal_pump_on=data_signal_pump_on,
            data_signal_pump_off=data_signal_pump_off,
        )


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    parametric_amplifier: TWPA,
    pump_frequency: ArrayLike,
    pump_power: ArrayLike,
    pump_on: bool = False,  # noqa: FBT001, FBT002
    probe_on: bool = False,  # noqa: FBT001, FBT002
    options: TWPATuneUpExperimentOptions | None = None,
) -> Experiment:
    """Creates a TWPA tune up Experiment.

    Arguments:
        qpu:
            The qpu consisting of the original pas and quantum operations.
        parametric_amplifier:
            The parametric amplifier to run the experiments on.
        pump_frequency:
            The frequencies to sweep over for each amplifier.
            It must be a list of lists of numbers or arrays.
        pump_power:
            The powers to sweep over for each parametric amplifier.
            It must be a list of lists of numbers or arrays.
        pump_on:
            Whether to turn on the pump tone.
        probe_on:
            Whether to turn on the probe tone.
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
        twpa = TWPA("twpa0")
        qpu = QPU(
            pas=[twpa],
            quantum_operations=TWPAOperations(),
        )
        create_experiment(
            qpu=qpu,
            parametric_amplifier=twpa,
            pump_power=np.linspace(0, 10, 501),
            pump_frequency=np.linspace(7.1e9, 7.6e9, 501),
            options=options,
        )
        ```
    """
    # Define the custom options for the experiment
    opts = TWPATuneUpExperimentOptions() if options is None else options
    pump_frequency = validation.validate_and_convert_sweeps_to_arrays(pump_frequency)
    pump_power = validation.validate_and_convert_sweeps_to_arrays(pump_power)

    # To measure both the incoherent and coherent part of the signal.
    if opts.use_probe_from_ppc or opts.do_snr:
        opts.acquisition_type = AcquisitionType.SPECTROSCOPY_PSD

    qop = qpu.quantum_operations
    frequency = SweepParameter(
        f"frequencies_{parametric_amplifier.uid}", pump_frequency
    )
    power = SweepParameter(f"powers_{parametric_amplifier.uid}", pump_power)
    if pump_on:
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
                parameter=power,
            ) as power:
                with dsl.sweep(
                    name=f"freq_{parametric_amplifier.uid}",
                    parameter=frequency,
                ) as frequency:
                    qop.set_readout_frequency(
                        parametric_amplifier,
                        parametric_amplifier.parameters.probe_frequency,
                    )
                    qop.set_pump_frequency(parametric_amplifier, frequency)
                    qop.set_pump_power(parametric_amplifier, power)
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
            qop.set_readout_frequency(
                parametric_amplifier, parametric_amplifier.parameters.probe_frequency
            )
            qop.twpa_acquire(
                parametric_amplifier,
                dsl.handles.result_handle(parametric_amplifier.uid),
            )
            qop.twpa_delay(parametric_amplifier, opts.spectroscopy_reset_delay)

    calibration = dsl.experiment_calibration()

    if opts.use_probe_from_ppc or not probe_on:
        qop.set_readout_amplitude(parametric_amplifier, 0)

    # TODO: Replace this by an operation to set the amplifier pump properties
    amplifier_pump = calibration[parametric_amplifier.signals["acquire"]].amplifier_pump
    amplifier_pump.pump_on = pump_on
    amplifier_pump.probe_on = opts.use_probe_from_ppc and probe_on
