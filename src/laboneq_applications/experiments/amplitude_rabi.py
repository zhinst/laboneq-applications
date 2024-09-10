"""This module defines the amplitude-rabi experiment.

In this experiment, we sweep the amplitude of a drive pulse on a given qubit transition
in order to determine the pulse amplitude that induces a rotation of pi.

The amplitude-rabi experiment has the following pulse sequence:

    qb --- [ prep transition ] --- [ x180_transition ] --- [ measure ]

If multiple qubits are passed to the `run` workflow, the above pulses are applied
in parallel on all the qubits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq.simple import Experiment, SweepParameter

from laboneq_applications import dsl
from laboneq_applications.analysis.amplitude_rabi import analysis_workflow
from laboneq_applications.experiments.options import (
    TuneupExperimentOptions,
    TuneUpWorkflowOptions,
)
from laboneq_applications.tasks import compile_experiment, run_experiment
from laboneq_applications.workflow import (
    comment,
    if_,
    task,
    workflow,
)

if TYPE_CHECKING:
    import uncertainties as unc
    from laboneq.dsl.session import Session

    from laboneq_applications.qpu_types import QPU
    from laboneq_applications.typing import Qubits, QubitSweepPoints


options = TuneUpWorkflowOptions


@workflow
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: Qubits,
    amplitudes: QubitSweepPoints,
    options: TuneUpWorkflowOptions | None = None,
) -> None:
    """The Amplitude Rabi Workflow.

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
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubits to run the experiments on. May be either a single
            qubit or a list of qubits.
        amplitudes:
            The amplitudes to sweep over for each qubit. If `qubits` is a
            single qubit, `amplitudes` must be a list of numbers or an array. Otherwise
            it must be a list of lists of numbers or arrays.
        options:
            The options for building the workflow.
            In addition to options from [WorkflowOptions], the following
            custom options are supported:
                - create_experiment: The options for creating the experiment.

    Returns:
        WorkflowBuilder:
            The builder of the experiment workflow.

    Example:
        ```python
        options = TuneUpExperimentWorkflowOptions()
        options.create_experiment.count = 10
        options.create_experiment.transition = "ge"
        qpu = QPU(
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            qop=TunableTransmonOperations(),
        )
        temp_qubits = qpu.copy_qubits()
        result = experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=temp_qubits,
            amplitudes=[
                np.linspace(0, 1, 11),
                np.linspace(0, 0.75, 11),
            ],
            options=options,
        ).run()
        ```
    """
    exp = create_experiment(
        qpu,
        qubits,
        amplitudes=amplitudes,
    )
    compiled_exp = compile_experiment(session, exp)
    _result = run_experiment(session, compiled_exp)
    with if_(options.do_analysis):
        analysis_results = analysis_workflow(_result, qubits, amplitudes)
        qubit_parameters = analysis_results.tasks["extract_qubit_parameters"].output
        with if_(options.update):
            update_qubits(qpu, qubit_parameters["new_parameter_values"])


@task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubits: Qubits,
    amplitudes: QubitSweepPoints,
    options: TuneupExperimentOptions | None = None,
) -> Experiment:
    """Creates an Amplitude Rabi experiment Workflow.

    Arguments:
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubits to run the experiments on. May be either a single
            qubit or a list of qubits.
        amplitudes:
            The amplitudes to sweep over for each qubit. If `qubits` is a
            single qubit, `amplitudes` must be a list of numbers or an array. Otherwise
            it must be a list of lists of numbers or arrays.
        options:
            The options for building the experiment.
            See [TuneupExperimentOptions] and [BaseExperimentOptions] for
            accepted options.
            Overwrites the options from [TuneupExperimentOptions] and
            [BaseExperimentOptions].

    Returns:
        experiment:
            The generated LabOne Q experiment instance to be compiled and executed.

    Raises:
        ValueError:
            If the qubits and qubit_amplitudes are not of the same length.

        ValueError:
            If qubit_amplitudes is not a list of numbers when a single qubit is passed.

        ValueError:
            If qubit_amplitudes is not a list of lists of numbers.

    Example:
        ```python
        options = {
            "count": 10,
            "transition": "ge",
            "averaging_mode": "cyclic",
            "acquisition_type": "integration_trigger",
            "cal_traces": True,
        }
        options = TuneupExperimentOptions(**options)
        qpu = QPU(
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            qop=TunableTransmonOperations(),
        )
        temp_qubits = qpu.copy_qubits()
        create_experiment(
            qpu=qpu,
            qubits=temp_qubits,
            amplitudes=[
                np.linspace(0, 1, 11),
                np.linspace(0, 0.75, 11),
            ],
            options=options,
        )
        ```
    """
    # Define the custom options for the experiment
    opts = TuneupExperimentOptions() if options is None else options
    qubits, amplitudes = dsl.validation.validate_and_convert_qubits_sweeps(
        qubits, amplitudes
    )
    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        for q, q_amplitudes in zip(qubits, amplitudes):
            with dsl.sweep(
                name=f"amps_{q.uid}",
                parameter=SweepParameter(f"amplitude_{q.uid}", q_amplitudes),
            ) as amplitude:
                qpu.qop.prepare_state(q, opts.transition[0])
                qpu.qop.x180(q, amplitude=amplitude, transition=opts.transition)
                qpu.qop.measure(q, dsl.handles.result_handle(q.uid))
                qpu.qop.passive_reset(q)
            if opts.use_cal_traces:
                with dsl.section(
                    name=f"cal_{q.uid}",
                ):
                    for state in opts.cal_states:
                        qpu.qop.prepare_state(q, state)
                        qpu.qop.measure(
                            q,
                            dsl.handles.calibration_trace_handle(q.uid, state),
                        )
                        qpu.qop.passive_reset(q)


@task
def update_qubits(
    qpu: QPU,
    qubit_parameters: dict[
        str,
        dict[str, dict[str, int | float | unc.core.Variable | None]],
    ],
) -> None:
    """Updates the parameters of the qubits in the qpu.

    Args:
        qpu: the qpu containing the qubits to be updated
        qubit_parameters: qubit parameters and the new values to be updated.
            This  dictionary has the following form:
            ```python
            {
                q.uid: {
                    qb_param_name: qb_param_value
                    }
            }
            ```
    """
    for qid, params_dict in qubit_parameters.items():
        if len(params_dict) == 0:
            comment(
                f"{qid} could not be updated because its "
                f"pi- and pi/2-pulse amplitudes could not be extracted."
            )
    qpu.update_qubits(qubit_parameters)
