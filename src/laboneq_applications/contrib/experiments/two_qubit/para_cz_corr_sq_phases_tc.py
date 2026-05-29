# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Single-qubit phase correction after a parametric CZ gate.

For each qubit pair, a phase sweep is run twice in sequence:

  Section 0 — target qubit (source in |0⟩):
    target --- [ x90 ] --- [ cz ] --- [ rz(phase) ] --- [ x90 ] --- [ measure ]
    source ---            [ cz ]

  Section 1 — source qubit (target in |0⟩):
    target ---            [ cz ]
    source --- [ x90 ] --- [ cz ] --- [ rz(phase) ] --- [ x90 ] --- [ measure ]

The Ramsey fringe in each section gives the single-qubit phase accumulated by
that qubit during the CZ gate, from which the correction Rz angle is extracted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq import workflow
from laboneq.simple import (
    Experiment,
    SectionAlignment,
    SweepParameter,
    dsl,
)
from laboneq.workflow.tasks import (
    compile_experiment,
    run_experiment,
)

from laboneq_applications.contrib.analysis.two_qubit.para_cz_corr_sq_phases_tc import (
    analysis_workflow,
)
from laboneq_applications.core.validation import (
    validate_and_extract_edges_from_qubit_pairs,
    validate_parallel_two_qubit_experiment,
)
from laboneq_applications.experiments.options import (
    TuneupExperimentOptions,
    TuneUpWorkflowOptions,
)
from laboneq_applications.qpu_types.tunable_coupler import TunableCoupler
from laboneq_applications.qpu_types.tunable_transmon import (
    TunableTransmonQubitParameters,
)
from laboneq_applications.tasks.parameter_updating import (
    temporary_qpu,
    update_qpu,
)

if TYPE_CHECKING:
    from laboneq.dsl.session import Session

    from laboneq_applications.qpu_types import QPU
    from laboneq_applications.typing import QubitSweepPoints


@workflow.workflow(name="para_cz_corr_sq_phases_tc")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubit_pairs: list[list[str]],
    *,
    phases: QubitSweepPoints,
    temporary_parameters: dict[str, dict | TunableTransmonQubitParameters]
    | None = None,
    options: TuneUpWorkflowOptions | None = None,
) -> None:
    """Single-qubit phase correction workflow for parametric CZ calibration.

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
            The QPU consisting of the qubits, couplers, and quantum operations.
        qubit_pairs:
            The qubit pairs on which to run the experiment, passed as a list of
            ``[source_uid, target_uid]`` UID pairs.
        phases:
            RZ phase sweep points, per qubit pair.
        temporary_parameters:
            Temporary parameter overrides applied via [temporary_qpu]().
        options:
            The options for building the workflow, passed as an instance of
            [TuneUpWorkflowOptions].

    Returns:
        WorkflowBuilder:
            The builder for the experiment workflow.
    """
    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    exp = create_experiment(
        qpu=temp_qpu,
        qubit_pairs=qubit_pairs,
        phases=phases,
    )

    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)

    with workflow.if_(options.do_analysis):
        analysis_results = analysis_workflow(
            result=result,
            qpu=temp_qpu,
            qubit_pairs=qubit_pairs,
            phases=phases,
        )

        qubit_parameters = analysis_results.output
        with workflow.if_(options.update):
            update_qpu(qpu, qubit_parameters["new_parameter_values"])

    workflow.return_(result)


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubit_pairs: list[list[str]],
    phases: QubitSweepPoints,
    options: TuneupExperimentOptions | None = None,
) -> Experiment:
    """Create the single-qubit phase correction experiment.

    Arguments:
        qpu:
            The QPU consisting of the qubits, couplers, and quantum operations.
        qubit_pairs:
            The qubit pairs on which to run the experiment, as a list of
            ``[source_uid, target_uid]`` UID pairs.
        phases:
            RZ phase sweep points, per qubit pair.
        options:
            The options for building the experiment as an instance of
            [TuneupExperimentOptions].

    Returns:
        experiment:
            The generated LabOne Q experiment instance to be compiled and executed.
    """
    opts = TuneupExperimentOptions() if options is None else options

    qubits = validate_parallel_two_qubit_experiment(qpu, qubit_pairs)
    edges = validate_and_extract_edges_from_qubit_pairs(
        qpu,
        "cz",
        qubit_pairs,
        element_class=TunableCoupler,
    )

    for e in edges:
        dsl.add_quantum_elements([e.source_node, e.target_node, e.quantum_element])

    phase_sweep_pars = [
        SweepParameter(
            f"phase_{e.quantum_element.uid}",
            e_phases,
            axis_name=f"{e.quantum_element.uid} phase [rad]",
        )
        for e, e_phases in zip(edges, phases, strict=False)
    ]

    qop = qpu.quantum_operations
    max_measure_section_length = qop.measure_section_length(qubits)

    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        with dsl.sweep(name="phase_sweep", parameter=phase_sweep_pars):
            if opts.active_reset:
                qop.active_reset(
                    qubits,
                    active_reset_states=opts.active_reset_states,
                    number_resets=opts.active_reset_repetitions,
                    measure_section_length=max_measure_section_length,
                )

            # Section 0: target qubit — source stays in |0⟩
            with dsl.section(name="main_section_0", alignment=SectionAlignment.RIGHT):
                for e, phase in zip(edges, phase_sweep_pars, strict=False):
                    qop.x90(e.target_node)
                    qop.cz(e.source_node, e.target_node)
                    qop.rz(e.target_node, phase)
                    qop.x90(e.target_node)

                with dsl.section(
                    name="main_measure_0", alignment=SectionAlignment.RIGHT
                ):
                    for e in edges:
                        sec = qop.measure(
                            e.target_node,
                            dsl.handles.result_handle(e.target_node.uid),
                        )
                        sec.length = max_measure_section_length
                        qop.passive_reset(e.target_node)

            # Section 1: source qubit — target stays in |0⟩
            with dsl.section(name="main_section_1", alignment=SectionAlignment.RIGHT):
                for e, phase in zip(edges, phase_sweep_pars, strict=False):
                    qop.x90(e.source_node)
                    qop.cz(e.source_node, e.target_node)
                    qop.rz(e.source_node, phase)
                    qop.x90(e.source_node)

                with dsl.section(
                    name="main_measure_1", alignment=SectionAlignment.RIGHT
                ):
                    for e in edges:
                        sec = qop.measure(
                            e.source_node,
                            dsl.handles.result_handle(e.source_node.uid),
                        )
                        sec.length = max_measure_section_length
                        qop.passive_reset(e.source_node)

        if opts.use_cal_traces:
            qop.calibration_traces.omit_section(
                qubits=qubits,
                states=opts.cal_states,
                active_reset=opts.active_reset,
                active_reset_states=opts.active_reset_states,
                active_reset_repetitions=opts.active_reset_repetitions,
                measure_section_length=max_measure_section_length,
            )
