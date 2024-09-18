"""This module defines the fine amplitude of x180 through error amplification experiment.

In this experiment, we sweep the number of a x180 pulse on qubit
in order to determine the fine amplitude of x180 pulse

The error-amplification experiment has the following pulse sequence:

                                           |number of pulses|
    qb0 --- [ prepare transition ] --- [ x180 ] [ x180 ] [ x180 ]--- [ measure ]
    qb1 --- [ prepare transition ] --- [ x180 ] [ x180 ] [ x180 ]--- [ measure ]

If multiple qubits are passed to the `run` workflow, the above pulses are applied
in parallel on all the qubits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq.simple import Experiment, SweepParameter

from laboneq_applications.core import handles
from laboneq_applications.core.build_experiment import qubit_experiment
from laboneq_applications.core.quantum_operations import dsl
from laboneq_applications.experiments.options import (
    TuneupExperimentOptions,
    TuneUpWorkflowOptions,
)
from laboneq_applications.tasks import compile_experiment, run_experiment
from laboneq_applications.workflow import task, workflow

if TYPE_CHECKING:
    from laboneq.dsl.session import Session

    from laboneq_applications.qpu_types import QPU
    from laboneq_applications.typing import Qubits, QubitSweepPoints


options = TuneUpWorkflowOptions

@workflow
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: Qubits,
    iterations: QubitSweepPoints,
    options: TuneUpWorkflowOptions | None = None,
) -> None:
    """The Fine Amplitude Workflow  .

    The workflow consists of the following steps:

    - [create_experiment]()
    - [compile_experiment]()
    - [run_experiment]()

    Arguments:
        session:
            The connected session to use for running the experiment.
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubits to run the experiments on. May be either a single
            qubit or a list of qubits.
        iterations:
            The number of iterations sweeping over for each qubit. If `qubits` is a
            single qubit, `iterations` must be a list of numbers or an array. Otherwise
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
        qpu = QPU(
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            qop=TunableTransmonOperations(),
        )
        temp_qubits = qpu.copy_qubits()
        result = experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=temp_qubits,
            iterations=[
                np.arange(0, 21, 1),
                np.arange(0, 21, 1),
            ],
            options=options,
        ).run()
        ```
    """
    exp = create_experiment(
        qpu,
        qubits,
        iterations=iterations,
    )
    compiled_exp = compile_experiment(session, exp)
    _result = run_experiment(session, compiled_exp)



@task
@qubit_experiment
def create_experiment(
    qpu: QPU,
    qubits: Qubits,
    iterations: QubitSweepPoints,
    options: TuneupExperimentOptions | None = None,

) -> Experiment:
    """Creates an Fine Amplitude experiment Workflow.

    Arguments:
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubits to run the experiments on. May be either a single
            qubit or a list of qubits.
        iterations:
            The number of iterations sweeping over for each qubit. If `qubits` is a
            single qubit, `iterations` must be a list of numbers or an array. Otherwise
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
            iterations=[
                np.arange(0, 21, 1),
                np.arange(0, 21, 1),
            ],
            options=options,
        )
        ```
    """
    # Define the custom options for the experiment
    opts = TuneupExperimentOptions() if options is None else options
    qubits, iterations = dsl.validation.validate_and_convert_qubits_sweeps(
        qubits, iterations
    )
    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        for q, q_iterations in zip(qubits, iterations):
            _sweep = SweepParameter(f"iter_{q.uid}", q_iterations)
            with dsl.sweep(
                name=f"iteration_{q.uid}",
                parameter=_sweep,
            ):
                qpu.qop.prepare_state(q, opts.transition[0])
                with dsl.match(sweep_parameter=_sweep):
                    for i in _sweep.values:
                        with dsl.case(state = i):
                            for _ in range(i):
                                qpu.qop.x180(q,transition=opts.transition)
                qpu.qop.measure(q, handles.result_handle(q.uid))
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

