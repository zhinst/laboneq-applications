"""This module defines the DRAG quadrature-scaling calibration experiment.

In this experiment, we determine the quadrature scaling factor, beta, of a DRAG pulse,
which is optimal for cancelling dynamics phase errors that occur during the application
of the pulse. The DRAG drive pulse has the following form:

v(t) = i(t) + q(t),

where the quadrature component is give by the derivative of the in-phase component,
scaled by a scaling factor beta:

q(t) = beta * d(i(t)) / d(t)

In order to determine the optimal beta for compensating phase errors, we apply a pulse
sequence that is sensitive to phase errors and sweep the value of beta for all the
drive pulses in the sequence. In the experiment workflow defined in this file, we
refer to the beta parameter as a q-scaling.

The DRAG quadrature-scaling calibration experiment has the following pulse sequence:

    qb --- [ prep transition ] --- [ x90_transition ]
    --- [ y180_transition ] --- [ measure ]

    qb --- [ prep transition ] --- [ x90_transition ]
    --- [ my180_transition ] --- [ measure ]

If multiple qubits are passed to the `run` workflow, the above pulses are applied
in parallel on all the qubits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from laboneq import workflow
from laboneq.simple import AveragingMode, Experiment, SweepParameter
from laboneq.workflow.tasks import (
    compile_experiment,
    run_experiment,
)

from laboneq_applications import dsl
from laboneq_applications.analysis.drag_q_scaling import analysis_workflow
from laboneq_applications.experiments.options import (
    TuneupExperimentOptions,
    TuneUpWorkflowOptions,
)
from laboneq_applications.tasks.parameter_updating import (
    temporary_modify,
    update_qubits,
)

if TYPE_CHECKING:
    from laboneq.dsl.quantum import (
        TransmonParameters,
    )
    from laboneq.dsl.session import Session

    from laboneq_applications.qpu_types import QPU
    from laboneq_applications.typing import Qubits, QubitSweepPoints


@workflow.workflow(name="drag_q_scaling")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: Qubits,
    q_scalings: QubitSweepPoints,
    temporary_parameters: dict[str, dict | TransmonParameters] | None = None,
    options: TuneUpWorkflowOptions | None = None,
) -> None:
    """The DRAG quadrature-scaling calibration workflow.

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
        q_scalings:
            The DRAG quadrature scaling factors to sweep over for each qubit
            (see docstring at the top of the module). If `qubits` is a single qubit,
            `q_scalings` must be a list of numbers or an array. Otherwise it must be a
            list of lists of numbers or arrays.
        temporary_parameters:
            The temporary parameters to update the qubits with.
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
        options = experiment_workflow()
        options.count(10)
        options.transition("ge")
        qpu = QPU(
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            quantum_operations=TunableTransmonOperations(),
        )
        temp_qubits = qpu.copy_qubits()
        result = experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=temp_qubits,
            q_scalings=[
                np.linspace(-0.05, 0.05, 11),
                np.linspace(-0.04, 0.04, 11),
            ],
            options=options,
        ).run()
        ```
    """
    qubits = temporary_modify(qubits, temporary_parameters)
    exp = create_experiment(
        qpu,
        qubits,
        q_scalings=q_scalings,
    )
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)
    with workflow.if_(options.do_analysis):
        analysis_results = analysis_workflow(result, qubits, q_scalings)
        qubit_parameters = analysis_results.output
        with workflow.if_(options.update):
            update_qubits(qpu, qubit_parameters["new_parameter_values"])
    workflow.return_(result)


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubits: Qubits,
    q_scalings: QubitSweepPoints,
    options: TuneupExperimentOptions | None = None,
) -> Experiment:
    """Creates a DRAG quadrature-scaling calibration Experiment.

    Arguments:
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubits to run the experiments on. May be either a single
            qubit or a list of qubits.
        q_scalings:
            The DRAG quadrature scaling factors to sweep over for each qubit
            (see docstring at the top of the module). If `qubits` is a single qubit,
            `q_scalings` must be a list of numbers or an array. Otherwise it must be a
            list of lists of numbers or arrays.
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
            If the qubits and q_scalings are not of the same length.

        ValueError:
            If q_scalings is not a list of numbers when a single qubit is passed.

        ValueError:
            If q_scalings is not a list of lists of numbers.

        ValueError:
            If the experiment uses calibration traces and the averaging mode is
            sequential.

    Example:
        ```python
        options = TuneupExperimentOptions()
        options.count(10)
        options.transition("ge")
        qpu = QPU(
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            quantum_operations=TunableTransmonOperations(),
        )
        temp_qubits = qpu.copy_qubits()
        create_experiment(
            qpu=qpu,
            qubits=temp_qubits,
            q_scalings=[
                np.linspace(-0.05, 0.05, 11),
                np.linspace(-0.05, 0.05, 11),
            ],
            options=options,
        )
        ```
    """
    # Define the custom options for the experiment
    opts = TuneupExperimentOptions() if options is None else options
    qubits, q_scalings = dsl.validation.validate_and_convert_qubits_sweeps(
        qubits, q_scalings
    )
    if (
        opts.use_cal_traces
        and AveragingMode(opts.averaging_mode) == AveragingMode.SEQUENTIAL
    ):
        raise ValueError(
            "'AveragingMode.SEQUENTIAL' (or {AveragingMode.SEQUENTIAL}) cannot be used "
            "with calibration traces because the calibration traces are added "
            "outside the sweep."
        )

    max_measure_section_length = qpu.measure_section_length(qubits)
    qop = qpu.quantum_operations
    pulse_ids = ["xx", "xy", "xmy"]
    ops_ids = ["x180", "y180", "y180"]
    phase_overrides = [0.0, np.pi / 2, -np.pi / 2]
    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        for q, q_scales in zip(qubits, q_scalings):
            with dsl.sweep(
                name=f"q_scalings_{q.uid}",
                parameter=SweepParameter(f"beta_{q.uid}", q_scales),
            ) as beta:
                for i, op_id in enumerate(ops_ids):
                    pulse_id = pulse_ids[i]
                    phase = phase_overrides[i]
                    qop.prepare_state(q, opts.transition[0])
                    qop.x90(q, pulse={"beta": beta}, transition=opts.transition)
                    qop[op_id](
                        q, pulse={"beta": beta}, phase=phase, transition=opts.transition
                    )
                    sec = qop.measure(
                        q, dsl.handles.result_handle(q.uid, suffix=pulse_id)
                    )
                    # we fix the length of the measure section to the longest section
                    # among the qubits to allow the qubits to have different readout
                    # and/or integration lengths.
                    sec.length = max_measure_section_length
                    qop.passive_reset(q)
            if opts.use_cal_traces:
                qop.calibration_traces(q, states=opts.cal_states)
