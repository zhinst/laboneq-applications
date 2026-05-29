# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the parametric CZ frequency vs phase experiment.

This is a characterization experiment for CPHASE gate calibration.

In this experiment, we sweep the frequency of a CZ gate (outer loop) and the
phase of an RZ gate (inner loop) on a given qubit pair in order to determine
the conditional phase accumulated during the CZ gate.

The experiment has the following pulse sequence:

    q0 --- [ x180 ] --- [ cz(freq) ] --- [ rz(phase) ] --- [ measure ]
    q1 --- [ x180 ] --- [ cz(freq) ] --- [           ] --- [ measure ]

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

from laboneq_applications.contrib.analysis.two_qubit.para_cz_freq_vs_phase import (
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


@workflow.workflow(name="para_cz_freq_vs_phase")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubit_pairs: list[list[str]],
    *,
    frequencies: QubitSweepPoints,
    phases: QubitSweepPoints,
    temporary_parameters: dict[str, dict | TunableTransmonQubitParameters]
    | None = None,
    options: TuneUpWorkflowOptions | None = None,
) -> None:
    """The parametric CZ frequency vs RZ phase workflow.

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
        frequencies:
            CZ gate frequencies, swept in the outer loop (per qubit pair).
        phases:
            RZ gate phases, swept in the inner loop (per qubit pair).
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
        frequencies=frequencies,
        phases=phases,
    )

    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)

    with workflow.if_(options.do_analysis):
        analysis_results = analysis_workflow(
            result=result,
            qpu=temp_qpu,
            qubit_pairs=qubit_pairs,
            frequencies=frequencies,
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
    frequencies: QubitSweepPoints,
    phases: QubitSweepPoints,
    options: TuneupExperimentOptions | None = None,
) -> Experiment:
    """Create the parametric CZ frequency vs RZ phase experiment.

    Arguments:
        qpu:
            The QPU consisting of the qubits, couplers, and quantum operations.
        qubit_pairs:
            The qubit pairs on which to run the experiment, as a list of
            ``[source_uid, target_uid]`` UID pairs.
        frequencies:
            CZ gate frequencies, per qubit pair.
        phases:
            RZ gate phases, per qubit pair.
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

    frequency_sweep_pars = [
        SweepParameter(
            f"frequency_{e.quantum_element.uid}",
            e_frequencies,
            axis_name=f"{e.quantum_element.uid} frequency [Hz]",
        )
        for e, e_frequencies in zip(edges, frequencies, strict=False)
    ]

    phase_sweep_pars = [
        SweepParameter(
            f"phase_{e.target_node.uid}",
            e_phases,
            axis_name=f"{e.target_node.uid} phase [rad]",
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
            name="frequency_sweep",
            parameter=frequency_sweep_pars,
        ):
            with dsl.sweep(
                name="phase_sweep",
                parameter=phase_sweep_pars,
            ):
                if opts.active_reset:
                    qop.active_reset(
                        qubits,
                        active_reset_states=opts.active_reset_states,
                        number_resets=opts.active_reset_repetitions,
                        measure_section_length=max_measure_section_length,
                    )

                with dsl.section(
                    name="main_section_0", alignment=SectionAlignment.RIGHT
                ):  # source qubit in state 0
                    for e, freq, phase in zip(
                        edges, frequency_sweep_pars, phase_sweep_pars, strict=False
                    ):
                        qop.x90(e.target_node)
                        qop.cz(e.source_node, e.target_node, frequency=freq)
                        qop.rz(e.target_node, phase)
                        qop.x90(e.target_node)

                    with dsl.section(
                        name="main_measure_0", alignment=SectionAlignment.RIGHT
                    ):
                        for e in edges:
                            sec = qop.measure(
                                e.target_node,
                                dsl.handles.result_handle(e.target_node.uid + "_0"),
                            )
                            sec.length = max_measure_section_length
                            qop.passive_reset(e.target_node)

                with dsl.section(
                    name="main_section_1", alignment=SectionAlignment.RIGHT
                ):  # source qubit in state 1
                    for e, freq, phase in zip(
                        edges, frequency_sweep_pars, phase_sweep_pars, strict=False
                    ):
                        qop.x180(e.source_node)

                        qop.x90(e.target_node)
                        qop.cz(e.source_node, e.target_node, frequency=freq)
                        qop.rz(e.target_node, phase)
                        qop.x90(e.target_node)

                    with dsl.section(
                        name="main_measure_1", alignment=SectionAlignment.RIGHT
                    ):
                        for e in edges:
                            sec = qop.measure(
                                e.target_node,
                                dsl.handles.result_handle(e.target_node.uid + "_1"),
                            )
                            sec.length = max_measure_section_length
                            qop.passive_reset(e.target_node)

            if opts.use_cal_traces:
                qop.calibration_traces.omit_section(
                    qubits=qubits,
                    states=opts.cal_states,
                    active_reset=opts.active_reset,
                    active_reset_states=opts.active_reset_states,
                    active_reset_repetitions=opts.active_reset_repetitions,
                    measure_section_length=max_measure_section_length,
                )
