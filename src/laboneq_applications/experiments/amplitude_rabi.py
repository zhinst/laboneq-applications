"""This module defines the amplitude-rabi experiment.

In this experiment, we sweep the amplitude of a drive pulse on a given qubit transition
in order to determine the pulse amplitude that induces a rotation of pi.

The amplitude-rabi experiment has the following pulse sequence:

    qb --- [ prep transition ] --- [ x180_transition ] --- [ measure ]

If multiple qubits are passed to the `run` taskbook, the above pulses are applied
in parallel on all the qubits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq.simple import Experiment, SweepParameter

from laboneq_applications.core import handles
from laboneq_applications.core.build_experiment import qubit_experiment
from laboneq_applications.core.options import (
    TuneupExperimentOptions,
)
from laboneq_applications.core.quantum_operations import QuantumOperations, dsl
from laboneq_applications.core.validation import validate_and_convert_qubits_sweeps
from laboneq_applications.tasks import compile_experiment, run_experiment
from laboneq_applications.workflow import TuneUpTaskBookOptions, task, taskbook

if TYPE_CHECKING:
    from laboneq.dsl.session import Session

    from laboneq_applications.typing import Qubits, QubitSweepPoints
    from laboneq_applications.workflow.taskbook import TaskBook


options = TuneUpTaskBookOptions


@taskbook
def run(
    session: Session,
    qop: QuantumOperations,
    qubits: Qubits,
    amplitudes: QubitSweepPoints,
    options: TuneUpTaskBookOptions | None = None,
) -> TaskBook:
    """The Amplitude Rabi TaskBook.

    The taskbook consists of the following steps:

    - [create_experiment]()
    - [compile_experiment]()
    - [run_experiment]()

    Arguments:
        session:
            The connected session to use for running the experiment.
        qop:
            The quantum operations to use when building the experiment.
        qubits:
            The qubits to run the experiments on. May be either a single
            qubit or a list of qubits.
        amplitudes:
            The amplitudes to sweep over for each qubit. If `qubits` is a
            single qubit, `amplitudes` must be a list of numbers or an array. Otherwise
            it must be a list of lists of numbers or arrays.
        options:
            The options for building the workflow.
            In addition to options from [TaskBookOptions], the following
            custom options are supported:
                - create_experiment: The options for creating the experiment.

    Returns:
        result:
            The result of the taskbook.

    Example:
        ```python
        options = TuneUpTaskBookOptions()
        options.create_experiment.count = 10
        options.create_experiment.transition = "ge"
        result = run(
            session=session,
            qop=qop,
            qubits=q0,
            amplitudes=[0.1, 0.2, 0.3],
            options=options,
        )
        ```
    """
    exp = create_experiment(
        qop,
        qubits,
        amplitudes=amplitudes,
    )
    compiled_exp = compile_experiment(session, exp)
    _result = run_experiment(session, compiled_exp)


@task
@qubit_experiment
def create_experiment(
    qop: QuantumOperations,
    qubits: Qubits,
    amplitudes: QubitSweepPoints,
    options: TuneupExperimentOptions | None = None,
) -> Experiment:
    """Creates an Amplitude Rabi Experiment.

    Arguments:
        qop:
            The quantum operations to use when building the experiment.
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
            "cal_traces": True
        }
        options = TuneupExperimentOptions(**options)
        create_experiment(
            qop=TunableTransmonOperations(),
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            amplitudes=[[0.1, 0.5, 1], [0.1, 0.5, 1]],
            options=options,
        )
        ```
    """
    # Define the custom options for the experiment
    opts = TuneupExperimentOptions() if options is None else options
    qubits, amplitudes = validate_and_convert_qubits_sweeps(qubits, amplitudes)
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
                qop.prepare_state(q, opts.transition[0])
                qop.x180(q, amplitude=amplitude, transition=opts.transition)
                qop.measure(q, handles.result_handle(q.uid))
                qop.passive_reset(q)
            if opts.use_cal_traces:
                with dsl.section(
                    name=f"cal_{q.uid}",
                ):
                    for state in opts.cal_states:
                        qop.prepare_state(q, state)
                        qop.measure(q, handles.calibration_trace_handle(q.uid, state))
                        qop.passive_reset(q)
