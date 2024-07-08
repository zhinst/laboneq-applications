"""Tasks for generating experiments."""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq.simple import Experiment, SweepParameter

from laboneq_applications.core import handles
from laboneq_applications.core.build_experiment import qubit_experiment
from laboneq_applications.core.options import (
    TaskBookOptions,
    TuneupExperimentOptions,
)
from laboneq_applications.core.quantum_operations import QuantumOperations, dsl
from laboneq_applications.core.validation import validate_and_convert_qubits_sweeps
from laboneq_applications.tasks import compile_experiment, run_experiment
from laboneq_applications.workflow import task, taskbook

if TYPE_CHECKING:
    from collections.abc import Sequence

    from laboneq.dsl.quantum.quantum_element import QuantumElement
    from laboneq.dsl.session import Session
    from numpy import ndarray

    from laboneq_applications.workflow.taskbook import TaskBook


class TuneUpTaskBookOptions(TaskBookOptions):
    """Option class for tune-up taskbook.

    Attributes:
        create_experiment (TuneupExperimentOptions):
            The options for creating the experiment.
            Default: TuneupExperimentOptions().
    """

    create_experiment: TuneupExperimentOptions = TuneupExperimentOptions()


@taskbook
def amplitude_rabi(
    session: Session,
    qop: QuantumOperations,
    qubits: QuantumElement | Sequence[QuantumElement],
    amplitudes: Sequence[float] | Sequence[Sequence[float] | ndarray] | ndarray,
    options: TuneUpTaskBookOptions | None = None,
) -> TaskBook:
    """Amplitude Rabi Experiment as a TaskBook.

    The amplitude Rabi TaskBook consists of the following steps:

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
        options = amplitude_rabi.options()
        options.create_experiment.count = 10
        options.create_experiment.transition = "ge"
        result = amplitude_rabi(
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
    qubits: QuantumElement | Sequence[QuantumElement],
    amplitudes: Sequence[float] | Sequence[Sequence[float] | ndarray] | ndarray,
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
        amplitude_rabi(
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
