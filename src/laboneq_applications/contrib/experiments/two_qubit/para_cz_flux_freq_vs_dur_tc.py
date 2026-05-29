# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the flux frequency vs flux duration experiment.

This is a characterization experiment for CPHASE gate calibration.

In this experiment, we sweep the frequency and duration of a flux pulse on a given
coupler in order to determine a good starting point for the CPHASE pulse. The
amplitude is fixed.

The experiment has the following pulse sequence:

    q0 --- [ x180_transition ] --- [                           ] --- [ measure ]
    c01--- [                 ] --- [ flux(frequency, duration) ] ---
    q1 --- [ x180_transition ] --- [                           ] --- [ measure ]

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

from laboneq_applications.contrib.analysis.two_qubit.para_cz_flux_freq_vs_dur_tc import (  # noqa: E501
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


@workflow.workflow(name="para_cz_flux_freq_vs_dur_tc")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubit_pairs: list[list[str]],
    *,
    frequencies: QubitSweepPoints,
    durations: QubitSweepPoints,
    temporary_parameters: dict[str, dict | TunableTransmonQubitParameters]
    | None = None,
    options: TuneUpWorkflowOptions | None = None,
) -> None:
    """The parametric CZ flux frequency vs duration calibration workflow.

    Sweeps the frequency and duration of a flux pulse on the tunable coupler (at
    fixed amplitude), with both qubits prepared in |1>, to map the |11>↔|20>
    chevron in (frequency, duration) space. The analysis fits each row with a
    cosine and a parabola across rows to extract the optimal flux frequency
    f_opt and pulse length 1/nu_min for a 2pi CZ gate.

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
            Frequencies of the coupler flux pulse, swept in the outer loop
            (per qubit pair).
        durations:
            Durations of the coupler flux pulse, swept in the inner loop
            (per qubit pair).
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
        durations=durations,
    )

    compiled_exp = compile_experiment(session, exp)

    result = run_experiment(session, compiled_exp)

    with workflow.if_(options.do_analysis):
        analysis_results = analysis_workflow(
            result,
            temp_qpu,
            qubit_pairs,
            frequencies,
            durations,
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
    durations: QubitSweepPoints,
    options: TuneupExperimentOptions | None = None,
) -> Experiment:
    """Create the parametric-CZ frequency-vs-duration experiment.

    Arguments:
        qpu:
            The QPU consisting of the qubits, couplers, and quantum operations.
        qubit_pairs:
            The qubit pairs on which to run the experiment, as a list of
            ``[source_uid, target_uid]`` UID pairs.
        frequencies:
            Frequencies of the coupler flux pulse, per qubit pair.
        durations:
            Durations of the coupler flux pulse, per qubit pair.
        options:
            The options for building the experiment as an instance of
            [TuneupExperimentOptions].

    Returns:
        experiment:
            The compiled-experiment-ready [Experiment] object.

    Raises:
        ValueError:
            If qubits in different pairs share resources, or if the topology has
            no ``"cz"`` edge of type [TunableCoupler] between any of the pairs.
    """
    # Define the custom options for the experiment
    opts = TuneupExperimentOptions() if options is None else options

    # get all edges from for the qubit pair and perform checks
    qubits = validate_parallel_two_qubit_experiment(qpu, qubit_pairs)
    # get all edges between the qubit pairs
    edges = validate_and_extract_edges_from_qubit_pairs(
        qpu,
        "cz",
        qubit_pairs,
        element_class=TunableCoupler,
    )

    # add all quantum elements to the experiment
    for e in edges:
        dsl.add_quantum_elements([e.source_node, e.target_node, e.quantum_element])

    frequency_sweep_pars = [
        SweepParameter(
            f"frequency_{e.quantum_element.uid}",
            e_frequencies,
            axis_name=f"{e.quantum_element.uid} frequency []",
        )
        for e, e_frequencies in zip(edges, frequencies, strict=False)
    ]

    duration_sweep_pars = [
        SweepParameter(
            f"duration_{e.quantum_element.uid}",
            e_durations,
            axis_name=f"{e.quantum_element.uid} duration [s]",
        )
        for e, e_durations in zip(edges, durations, strict=False)
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
            name="pulse_frequency_sweep",
            parameter=frequency_sweep_pars,
        ):
            with dsl.sweep(
                name="pulse_duration_sweep",
                parameter=duration_sweep_pars,
            ):
                if opts.active_reset:
                    qop.active_reset(
                        qubits,
                        active_reset_states=opts.active_reset_states,
                        number_resets=opts.active_reset_repetitions,
                        measure_section_length=max_measure_section_length,
                    )
                ## apply uncalibrated CZ gate
                with dsl.section(
                    name="Main_CZ_calibration", alignment=SectionAlignment.RIGHT
                ):
                    for e, freq, duration in zip(
                        edges, frequency_sweep_pars, duration_sweep_pars, strict=False
                    ):
                        # excite to |1>
                        qop.x180(e.source_node)
                        qop.x180(e.target_node)
                        # TODO: Do we want phase to be zero at this stage?
                        cz_parameters = {"frequency": freq, "length": duration}
                        qop.cz(e.source_node, e.target_node, **cz_parameters)

                # Measure only the target qubit: the |11> chevron oscillation
                # is observed on the target.
                with dsl.section(name="main_measure", alignment=SectionAlignment.RIGHT):
                    for e in edges:
                        sec = qop.measure(
                            e.target_node, dsl.handles.result_handle(e.target_node.uid)
                        )
                        # Fix the length of the measure section
                        sec.length = max_measure_section_length
                        qop.passive_reset(e.target_node)

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
