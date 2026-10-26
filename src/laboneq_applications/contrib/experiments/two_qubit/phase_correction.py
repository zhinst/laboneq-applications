# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines a CZ phase-correction experiment on a pair of qubits.

Each pair, written here as q0 and q1, is coupled through a tunable coupler (TC).

Check the phase correction for both qubits involved in CZ in a Ramsey-type experiment.
Sequence is applied to both qubits one after the other. Second pi/2 is played with a
phase shift that is being swept.

The Cz experiment has the following pulse sequence:

    q0 ------------------------------------------[measure][reset]---
    q1 -[pi/2][modulated ffp][pi/2 phase shifted][measure][reset]---
    tc -------[     ffp     ]--------------------[discharge]--------

followed by:

    q0 -[pi/2]-----------------------------------[measure][reset]---
    q1 -------[modulated ffp][pi/2 phase shifted][measure][reset]---
    tc -------[     ffp     ]--------------------[discharge]--------
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

from laboneq_applications.contrib.analysis.two_qubit.phase_correction import (
    analysis_workflow,
)
from laboneq_applications.contrib.experiments.two_qubit.options import (
    TuneupExperiment2QOptions,
)
from laboneq_applications.core.validation import (
    validate_flux_driven_edges,
    validate_parallel_two_qubit_experiment,
)
from laboneq_applications.experiments.options import (
    TuneUpWorkflowOptions,
)
from laboneq_applications.qpu_types.tunable_transmon import (
    TunableTransmonQubitParameters,
)
from laboneq_applications.tasks import extract_edges_from_qubit_pairs
from laboneq_applications.tasks.parameter_updating import (
    temporary_qpu,
    update_qpu_edges,
)

if TYPE_CHECKING:
    from laboneq.dsl.quantum import QuantumElement
    from laboneq.dsl.session import Session
    from numpy.typing import ArrayLike

    from laboneq_applications.qpu_types import QPU
    from laboneq_applications.typing import QubitSweepPoints


@workflow.workflow(name="phase_correction")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubit_pairs: ArrayLike[ArrayLike[QuantumElement, QuantumElement]],
    phases: QubitSweepPoints,
    temporary_parameters: dict[str, dict | TunableTransmonQubitParameters]
    | None = None,
    options: TuneUpWorkflowOptions | None = None,
) -> None:
    """The phase correction Workflow.

    The workflow consists of the following steps:

    - [create_experiment][]
    - [compile_experiment][]
    - [run_experiment][]
    - [analysis_workflow][]
    - [update_qpu][]

    Arguments:
        session:
            The connected session to use for running the experiment.
        qpu:
            The qpu consisting of the original qubits, couplers and quantum operations.
        qubit_pairs:
            Pairs of qubit to run the experiment on.
        phases:
            The phase shift to apply on the qubit that the phase is being measured on.
            Must be a list of numbers or an array.
        temporary_parameters:
            The temporary parameters to update the qubits and couplers with.
        options:
            The options for building the workflow.
            In addition to options from [WorkflowOptions][]

    Returns:
        result:
            The result of the workflow.
    """
    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    exp = create_experiment(
        temp_qpu,
        qubit_pairs,
        phases,
    )
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)
    with workflow.if_(options.do_analysis):
        analysis_result = analysis_workflow(
            result,
            temp_qpu,
            qubit_pairs,
            phases,
        )
        edge_parameters = analysis_result.output
        with workflow.if_(options.update):
            update_qpu_edges(qpu, edge_parameters["new_parameter_values"])

    workflow.return_(result)


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubit_pairs: list[list[str]],
    phases: QubitSweepPoints,
    options: TuneupExperiment2QOptions | None = None,
) -> Experiment:
    """Creates phase correction experiment workflow.

    Arguments:
        qpu:
            The qpu consisting of the original quibts, couplers, and quantum operations.
        qubit_pairs:
            Pairs of qubits to run the experiment on.
        phases:
            The phase shift to apply on the qubit that the phase is being measured on.
            Must be a list of numbers or an array.
        options:
            The options for building the experiment.
            See [TuneupExperiment2QOptions][] and [TuneupExperimentOptions][] for
            accepted options.

    Returns:
        experiment:
            The generated LabOne Q experiment instance to be compiled and executed.

    """
    opts = TuneupExperiment2QOptions() if options is None else options

    # get all edges from for the qubit pair and perform checks
    qubits = validate_parallel_two_qubit_experiment(qpu, qubit_pairs)

    # get all edges between the qubit pairs
    edges = extract_edges_from_qubit_pairs(
        qpu,
        qubit_pairs,
        tag="cz",
    )
    validate_flux_driven_edges(edges)

    # add all quantum elements to the experiment
    for e in edges:
        dsl.add_quantum_elements([e.source_node, e.target_node, e.quantum_element])

    phase_sweep_pars = [
        SweepParameter(
            f"x90_phases_{e.source_node.uid}",
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
        with dsl.sweep(
            name="phase_sweep",
            parameter=phase_sweep_pars,
            auto_chunking=opts.auto_chunking,
        ):
            # correct phase on control qubit
            for e, phase in zip(edges, phase_sweep_pars, strict=False):
                with dsl.section(name=f"phase_correction_control_{e.source_node.uid}"):
                    qop.x90(e.source_node)
                    qop.cz(e.source_node, e.target_node)
                    qop.x90(e.source_node, increment_oscillator_phase=phase)
            ## measure all qubits
            with dsl.section(
                name="main_measure_control", alignment=SectionAlignment.RIGHT
            ):
                for q in qubits:
                    sec = qop.measure(
                        q, dsl.handles.result_handle(q.uid, suffix="control")
                    )
                    # Fix the length of the measure section
                    sec.length = max_measure_section_length
                    qop.passive_reset(q)
            # correct phase on target qubit
            for e, phase in zip(edges, phase_sweep_pars, strict=False):
                with dsl.section(name=f"phase_correction_target_{e.target_node.uid}"):
                    qop.x90(e.target_node)
                    qop.cz(e.source_node, e.target_node)
                    qop.x90(e.target_node, increment_oscillator_phase=phase)
            ## measure all qubits
            with dsl.section(
                name="main_measure_target", alignment=SectionAlignment.RIGHT
            ):
                for q in qubits:
                    sec = qop.measure(
                        q, dsl.handles.result_handle(q.uid, suffix="target")
                    )
                    # Fix the length of the measure section
                    sec.length = max_measure_section_length
                    qop.passive_reset(q)

        ## extract calibration traces and reset qubit if selected
        if opts.use_cal_traces:
            qop.calibration_traces.omit_section(
                qubits=qubits,
                states=opts.cal_states,
                active_reset=opts.active_reset,
                active_reset_states=opts.active_reset_states,
                active_reset_repetitions=opts.active_reset_repetitions,
                measure_section_length=max_measure_section_length,
            )
