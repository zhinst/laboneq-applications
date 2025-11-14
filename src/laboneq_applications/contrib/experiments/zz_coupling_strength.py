# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Cross ZZ Echo Ramsey experiment used to characterize residual ZZ interaction.

In this experiment, we sweep the flux bias applied to the coupler. At each bias point we
perform a Cross ZZ Echo Ramsey experiment between the two qubits to determine the
frequency shift of qubits[0] when qubits[1] is in the excited vs ground state.


The decoupling experiment has the following pulse sequence:
[tau] and the static dc bias are swept

         q0 -[X_pi/2]-[tau]-[X_pi]-[tau]-[Y_pi/2]-[measure][passive reset]
         q1 ----------------[X_pi]-------[X_pi]---[measure][passive reset]
    coupler ----------------------------------------------------------- (static dc bias)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq import workflow
from laboneq.simple import (
    AveragingMode,
    Experiment,
    SectionAlignment,
    SweepParameter,
    dsl,
)
from laboneq.workflow.tasks import (
    compile_experiment,
    run_experiment,
)

from laboneq_applications.contrib.analysis.zz_coupling_strength import analysis_workflow
from laboneq_applications.core.validation import (
    validate_and_extract_edges_from_qubit_pairs,
    validate_parallel_two_qubit_experiment,
)
from laboneq_applications.experiments.options import (
    TuneupExperimentOptions,
    TuneUpWorkflowOptions,
)
from laboneq_applications.qpu_types.tunable_coupler import (
    TunableCoupler,
)
from laboneq_applications.qpu_types.tunable_transmon import (
    TunableTransmonQubitParameters,
)
from laboneq_applications.tasks import (
    temporary_qpu,
    update_qpu,
)

if TYPE_CHECKING:
    from laboneq.dsl.session import Session

    from laboneq_applications.qpu_types import QPU
    from laboneq_applications.typing import QubitSweepPoints


@workflow.workflow(name="zz_coupling_strength_exp")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubit_pairs: list[list[str]],
    biases: QubitSweepPoints,
    delays: QubitSweepPoints,
    temporary_parameters: dict[str, dict | TunableTransmonQubitParameters]
    | None = None,
    options: TuneUpWorkflowOptions | None = None,
) -> None:
    """The ZZ coupling strength workflow.

    The workflow consists of the following steps:

    - [temporary_qpu]()
    - [create_experiment]()
    - [compile_experiment]()
    - [run_experiment]()
    - [analysis_workflow]()
    - [update_qpu]()

    Arguments:
        session:
            The connected session to use for running the experiment.
        qpu:
            The QPU consisting of the original qubits, coupler, and quantum operations.
        qubit_pairs:
            The two qubits on which to run the experiment.
        biases:
            The DC voltage biases applied to the coupler. This is the inner sweep
            parameter (fast axis).
        delays:
            The delays (in seconds) of the time between the first X90 and the X180,
            and the X180 and the Y90. This is the outer sweep parameter (slow axis).
        temporary_parameters:
            The temporary parameters with which to update the QPU.
        options:
            The options for building the workflow.
            In addition to options from [WorkflowOptions]

    Returns:
        result:
            The result of the workflow.
    """
    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    exp = create_experiment(
        temp_qpu,
        qubit_pairs,
        biases,
        delays,
    )
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)

    with workflow.if_(options.do_analysis):
        analysis_result = analysis_workflow(
            result,
            temp_qpu,
            qubit_pairs,
            biases,
            delays,
        )
        edge_parameters = analysis_result.output
        with workflow.if_(options.update):
            update_qpu(qpu, edge_parameters["new_parameter_values"])
    workflow.return_(result)


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubit_pairs: list[list[str]],
    biases: QubitSweepPoints,
    delays: QubitSweepPoints,
    options: TuneupExperimentOptions | None = None,
) -> Experiment:
    """Creates a ZZ coupling strength experiment.

    Arguments:
        qpu:
            The QPU consisting of the original qubits, coupler, and quantum operations.
        qubit_pairs:
            A list of UIDs of qubit pairs with the format
            [["q0", "q1"], ["q2", "q4"], ...]. The workflow will be executed in parallel
            on these pairs, throwing an error if there is a conflict of resources in the
            sequence, for example the same qubit UID appearing in multiple pairs.
        biases:
            Voltage offset to be applied to the flux channel connected to the tunable
            coupler of each pair.
        delays:
            Delays to be used in the Ramsey experiment executed to characterize the
            frequency shift between the two qubits.
        options:
            The options for building the experiment.
            See [TuneupExperimentOptions] for accepted options.

    Returns:
        experiment:
            The generated LabOne Q experiment instance to be compiled and executed.

    Raises:
        ValueError:
            If the qubits and delays and biases are not of the same length.

        ValueError:
            If qubits passed have conflicting resources, for example a qubit appears
            more than once per pair.

        ValueError:
            If one of the edges of `target_tag` is not of type `TunableCoupler`.

        ValueError:
            If the experiment uses calibration traces and the averaging mode is
            sequential.

    """
    opts = TuneupExperimentOptions() if options is None else options
    qubits = validate_parallel_two_qubit_experiment(qpu, qubit_pairs)
    # get all edges between the qubit pairs
    edges = validate_and_extract_edges_from_qubit_pairs(
        qpu,
        "coupler",
        qubit_pairs,
        # TODO: option instead?
        element_class=TunableCoupler,
    )

    # add all quantum elements to the experiment
    for e in edges:
        dsl.add_quantum_elements([e.source_node, e.target_node, e.quantum_element])

    if (
        opts.use_cal_traces
        and AveragingMode(opts.averaging_mode) == AveragingMode.SEQUENTIAL
    ):
        raise ValueError(
            "'AveragingMode.SEQUENTIAL' (or {AveragingMode.SEQUENTIAL}) cannot be used "
            "with calibration traces because the calibration traces are added "
            "outside the sweep."
        )

    bias_sweep_pars = [
        SweepParameter(
            f"bias_{e.quantum_element.uid}",
            e_biases,
            axis_name=f"{e.quantum_element.uid}",
        )
        for e, e_biases in zip(edges, biases)
    ]

    delay_sweep_pars = [
        SweepParameter(
            f"delay_{e.quantum_element.uid}",
            e_delays,
            axis_name=f"{e.quantum_element.uid}",
        )
        for e, e_delays in zip(edges, delays)
    ]

    qop = qpu.quantum_operations
    max_measure_section_length = qpu.measure_section_length(qubits)

    with dsl.sweep(
        name="coupler_bias_sweep",
        parameter=bias_sweep_pars,
    ):
        with dsl.acquire_loop_rt(
            count=opts.count,
            averaging_mode=opts.averaging_mode,
            acquisition_type=opts.acquisition_type,
            repetition_mode=opts.repetition_mode,
            repetition_time=opts.repetition_time,
            reset_oscillator_phase=opts.reset_oscillator_phase,
        ):
            with dsl.sweep(
                name="delays_sweep",
                parameter=delay_sweep_pars,
            ):
                with dsl.section(name="main", alignment=SectionAlignment.RIGHT):
                    for e, e_delays in zip(edges, delay_sweep_pars):
                        with dsl.section(
                            name=f"rotate1_{e.source_node}",
                            alignment=SectionAlignment.LEFT,
                        ):
                            qop.x90.omit_section(e.source_node)

                        with dsl.section(
                            name="delay_1", alignment=SectionAlignment.LEFT
                        ):
                            qop.delay(e.source_node, time=e_delays)
                            qop.delay(e.target_node, time=e_delays)

                        with dsl.section(
                            name="interact",
                            alignment=SectionAlignment.LEFT,
                        ):
                            qop.x180.omit_section(e.source_node)
                            qop.x180.omit_section(e.target_node)

                        with dsl.section(
                            name="delay_2", alignment=SectionAlignment.RIGHT
                        ):
                            qop.delay(e.source_node, time=e_delays)
                            qop.delay(e.target_node, time=e_delays)

                        with dsl.section(
                            name="rotate2",
                            alignment=SectionAlignment.RIGHT,
                        ):
                            qop.y90.omit_section(e.source_node)
                            qop.x180.omit_section(e.target_node)

                    with dsl.section(
                        name="main_measure", alignment=SectionAlignment.LEFT
                    ):
                        for q in qubits:
                            sec = qop.measure(q, dsl.handles.result_handle(q.uid))
                            # Fix the length of the measure section
                            sec.length = max_measure_section_length
                            qop.passive_reset(q)
            if opts.use_cal_traces:
                qop.calibration_traces.omit_section(
                    qubits=qubits,
                    states=opts.cal_states,
                    active_reset=opts.active_reset,
                    active_reset_states=opts.active_reset_states,
                    active_reset_repetitions=opts.active_reset_repetitions,
                    measure_section_length=max_measure_section_length,
                )

    calibration = dsl.experiment_calibration()
    for e, bias in zip(edges, bias_sweep_pars):
        calibration[e.quantum_element.signals["flux"]].voltage_offset = bias
