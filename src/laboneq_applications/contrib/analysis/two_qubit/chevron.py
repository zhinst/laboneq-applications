# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


"""This module defines the analysis for a CZ chevron experiment.

The experiment is defined in
`laboneq_applications.contrib.experiments.two_qubit.chevron`.

In this analysis, we first interpret the raw data into qubit population using
principal-component analysis, or rotation and projection onto the measured
calibration states. We then locate the population extremum of the resulting
two-dimensional map to obtain the flux-pulse duration and amplitude that
implement the CZ gate, and plot the chevron pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from laboneq import workflow
from laboneq.dsl.quantum import QPU
from laboneq.dsl.quantum.qpu_topology import TopologyEdge

from laboneq_applications.analysis.calibration_traces_rotation import (
    calculate_qubit_population_2d,
)
from laboneq_applications.analysis.options import (
    ExtractEdgeParametersOptions,
    PlotPopulationOptions,
    TuneUpAnalysisWorkflowOptions,
)
from laboneq_applications.analysis.plotting_helpers import (
    plot_data_2d,
    plot_raw_complex_data_2d,
)
from laboneq_applications.contrib.analysis.two_qubit.options import (
    FitData2QOptions,
)
from laboneq_applications.conversion import as_quantum_element_list
from laboneq_applications.core.validation import (
    validate_flux_driven_edges,
)
from laboneq_applications.tasks import (
    extract_edges_from_qubit_pairs,
    extract_nodes_from_edges,
)

if TYPE_CHECKING:
    import matplotlib as mpl
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults
    from numpy.typing import ArrayLike

    from laboneq_applications.typing import (
        QuantumElements,
        QubitSweepPoints,
    )


@workflow.workflow
def analysis_workflow(
    result: RunExperimentResults,
    qpu: QPU,
    qubit_pairs: ArrayLike[ArrayLike[str, str]],
    durations: QubitSweepPoints,
    amplitudes: QubitSweepPoints,
    options: TuneUpAnalysisWorkflowOptions | None = None,
) -> None:
    """The CZ chevron analysis workflow.

    The workflow consists of the following steps:

    - [extract_edges_from_qubit_pairs][]
    - [validate_flux_driven_edges][]
    - [extract_nodes_from_edges][]
    - [calculate_qubit_population][]
    - [fit_data][]
    - [extract_edge_parameters][]
    - [plot_raw_complex_data_2d][]
    - [plot_population2d][]

    Arguments:
        result:
            The experiment results returned by the `run_experiment` task.
        qpu:
            The quantum processing unit.
        qubit_pairs:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the result.
        amplitudes:
            The coupler or qubit amplitude values.
        durations:
            length of the pulse on the coupler or qubit.
        options:
            The options for building the workflow, passed as an instance of
                [TuneUpAnalysisWorkflowOptions][].
            In addition to options from [WorkflowOptions][], the following
            custom options are supported: `do_fitting`, `do_plotting`, and the options
            of the [TuneupAnalysisOptions][] class. See the docstring of
            [TuneUpAnalysisWorkflowOptions][] for more details.

    Returns:
        WorkflowBuilder:
            The builder for the analysis workflow.

    Example:
        ```python
        options = TuneUpAnalysisWorkflowOptions()
        result = analysis_workflow(
            results=results
            qubits=[q0, q1],
            amplitudes=[
                np.linspace(0, 1, 11),
                np.linspace(0, 0.75, 11),
            ],
            options=options,
        ).run()
        ```
    """
    # extract edges and sweep parameters from result
    edges = extract_edges_from_qubit_pairs(
        qpu,
        qubit_pairs,
        tag="cz",
    )
    validate_flux_driven_edges(edges)

    qubits_source = extract_nodes_from_edges(edges, "source")
    qubits_target = extract_nodes_from_edges(edges, "target")

    processed_data_dict = calculate_qubit_population_2d(
        qubits=qubits_source,
        result=result,
        sweep_points_1d=amplitudes,
        sweep_points_2d=durations,
    )
    processed_data_dict_target = calculate_qubit_population_2d(
        qubits=qubits_target,
        result=result,
        sweep_points_1d=amplitudes,
        sweep_points_2d=durations,
    )

    fit_results = fit_data(qubit_pairs, processed_data_dict_target)

    edges_parameters = extract_edge_parameters(
        edges, processed_data_dict_target, fit_results
    )

    with workflow.if_(options.do_plotting):
        with workflow.if_(options.do_raw_data_plotting):
            plot_raw_complex_data_2d(
                qubits_source,
                result,
                sweep_points_1d=amplitudes,
                sweep_points_2d=durations,
                label_sweep_points_1d="Pulse Amplitude, (a.u)",
                label_sweep_points_2d="Pulse Length, $L_{cz}$ (ns)",
                scaling_sweep_points_1d=1,
                scaling_sweep_points_2d=1e-9,
            )
            plot_raw_complex_data_2d(
                qubits_target,
                result,
                sweep_points_1d=amplitudes,
                sweep_points_2d=durations,
                label_sweep_points_1d="Pulse Amplitude, (a.u)",
                label_sweep_points_2d="Pulse Length, $L_{cz}$ (ns)",
                scaling_sweep_points_1d=1,
                scaling_sweep_points_2d=1e-9,
            )
        with workflow.if_(options.do_qubit_population_plotting):
            plot_population(qubits_source, processed_data_dict)
            plot_population(qubits_target, processed_data_dict_target)

    workflow.return_(edges_parameters)


@workflow.task
def fit_data(
    qubit_pairs: list[list[str]],
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    options: FitData2QOptions | None = None,
) -> dict[str, dict[str, float]]:
    """Extract the point of maximum interaction from the chevron pattern.

    Finds the (duration, amplitude) sweep point at which the target-qubit
    population is maximal.

    Arguments:
        qubit_pairs:
            The (source, target) qubit-UID pairs on which to run the task.
            The target UIDs must exist in `processed_data_dict`. Note that
            the task takes the qubit-pair UIDs rather than the topology
            edges, so that its input remains serializable by the logbook
            stores.
        processed_data_dict:
            The processed data dictionary containing the qubit population
            and the sweep points of the experiment.
        options:
            The options for building the workflow as an instance of
            [FitData2QOptions][]. See the docstrings of this class
            for more details.

    Returns:
        Dictionary with edge signatures as keys and dictionaries with the
            "duration" and "amplitude" values of the maximum-population point
            as values.
    """
    opts = FitData2QOptions() if options is None else options

    fit_results = {}
    if not opts.do_fitting:
        return fit_results

    for q_source, q_target in qubit_pairs:
        edge_signature = f"cz:{q_source}:{q_target}"
        data = processed_data_dict[q_target]["population"]
        amplitudes = processed_data_dict[q_target]["sweep_points_1d"]
        durations = processed_data_dict[q_target]["sweep_points_2d"]
        # WARNING: extract maximum point, assumption on results
        # population has shape (len(sweep_points_2d), len(sweep_points_1d)),
        # i.e. (durations, amplitudes), because the duration sweep is the
        # outermost one
        max_row, max_col = np.unravel_index(np.argmax(data, axis=None), data.shape)
        fit_results[edge_signature] = {
            "duration": durations[max_row],
            "amplitude": amplitudes[max_col],
        }

    return fit_results


@workflow.task
def plot_population(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    options: PlotPopulationOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Create the chevron population plots.

    Arguments:
        qubits:
            The qubits on which to run the task. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in
            `processed_data_dict`, `fit_results` and `qubit_parameters`.
        processed_data_dict:
            The processed data dictionary returned by `process_raw_data`.
        options:
            The options for processing the raw data.
            See [PlotPopulationOptions][], [TuneupExperimentOptions][] and
            [BaseExperimentOptions][] for accepted options.
            Overwrites the options from [PlotPopulationOptions][],
            [TuneupExperimentOptions][] and [BaseExperimentOptions][].

    Returns:
        Dictionary with qubit UIDs as keys and the figures for each qubit as values.

            If a qubit UID is not found in `fit_results`, the fit and the textbox with
            the extracted qubit parameters are not plotted.
    """
    opts = PlotPopulationOptions() if options is None else options
    qubits = as_quantum_element_list(qubits)

    figures = {}
    for q in qubits:
        num_cal_traces = processed_data_dict[q.uid]["num_cal_traces"]

        echo_pulse_length = (
            q.parameters.ef_drive_length
            if "f" in opts.transition
            else q.parameters.ge_drive_length
        )

        amplitudes = processed_data_dict[q.uid]["sweep_points_1d"]
        durations = processed_data_dict[q.uid]["sweep_points_2d"] + echo_pulse_length
        data = processed_data_dict[q.uid][
            "population" if opts.do_rotation else "data_raw"
        ]

        fig, axs = plt.subplots()
        # the data is indexed by duration first, so it has to be transposed:
        # plot_data_2d expects z_values shaped len y_values by len x_values
        fig, axs = plot_data_2d(
            x_values=durations,
            y_values=amplitudes,
            z_values=data.T,
            label_x_values="Duration of gate, $\\tau$ ($\\mu$s)",
            label_y_values="Flux amplitude, $V_{gate}$ (V)",
            label_z_values="Principal Component (a.u)"
            if (num_cal_traces == 0 or opts.do_pca)
            else f"$|{opts.cal_states[-1]}\\rangle$-State Population",
            scaling_x_values=1e6,
            figure=fig,
            plot_title=f" chevron {q.uid}",
            axis=axs,
            close_figures=opts.close_figures,
        )

        if opts.save_figures:
            workflow.save_artifact(f"ZZ_coupling_strength_{q.uid}", fig)

        figures[q.uid] = fig

    return figures


@workflow.task(save=False)
def extract_edge_parameters(
    edges: list[TopologyEdge],
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    fit_results: dict[str, dict[str, float]],
    options: ExtractEdgeParametersOptions | None = None,
) -> dict[str, dict[tuple[str, str, str], dict[str, int | float | None]]]:
    """Extract the edge parameters from the fit results.

    The (duration, amplitude) point of maximum interaction found by `fit_data`
    becomes the starting point for the subsequent fine calibration of the CZ
    gate. The duration goes into `coupler_pulse`, because the gate takes its
    length from there, and the amplitude into the pulse that drives the gate.

    Arguments:
        edges: The array of topology edges.
        processed_data_dict: The processed data dictionary.
        fit_results: The fit-results dictionary returned by `fit_data`.
        options:
            The options for extracting the qubit parameters.
            See [ExtractEdgeParametersOptions][] for accepted options.

    Returns:
        Dictionary with extracted edge parameters and the previous values for
            those edge parameters. The keys of the inner dictionaries are the edge
            keys `(tag, source node UID, target node UID)`, as accepted by
            `update_qpu`. The dictionary has the following form:

            ```python
            {
                "new_parameter_values": {
                    "tag:source_uid:target_uid": {
                        edge_param_name: edge_param_value
                    },
                }
                "old_parameter_values": {
                    "tag:source_uid:target_uid": {
                        edge_param_name: edge_param_value
                    },
                }
            }
            ```

            If the `do_fitting` option is False, the `new_parameter_values` are not
            extracted and the function only returns the `old_parameter_values`.
            If an edge is not found in `fit_results`, the `new_parameter_values`
            entry for that edge is left empty.
    """
    edge_parameters = {
        "old_parameter_values": {
            f"{e.tag}:{e.source_node.uid}:{e.target_node.uid}": {} for e in edges
        },
        "new_parameter_values": {
            f"{e.tag}:{e.source_node.uid}:{e.target_node.uid}": {} for e in edges
        },
    }

    for e in edges:
        edge_key = f"{e.tag}:{e.source_node.uid}:{e.target_node.uid}"
        edge_signature = edge_key

        # the gate drives the qubit flux line when the edge defines a control
        # pulse, and the coupler otherwise
        qubit_driven = e.parameters.control_pulse is not None

        # Store the previous pulse parameters of the edge
        old_coupler_pulse = dict(e.parameters.coupler_pulse)
        old_pulses = {"coupler_pulse": old_coupler_pulse}
        if qubit_driven:
            old_control_pulse = dict(e.parameters.control_pulse)
            old_pulses["control_pulse"] = old_control_pulse
        edge_parameters["old_parameter_values"][edge_key] = old_pulses

        if edge_signature not in fit_results:
            workflow.comment(
                f"No fit results for {edge_signature}, "
                "skipping new parameter extraction."
            )
            continue

        # extract the point of maximum interaction; the gate reads its length from
        # the coupler pulse and its amplitude from the pulse that drives it
        new_coupler_pulse = dict(old_coupler_pulse)
        new_coupler_pulse["length"] = fit_results[edge_signature]["duration"]
        new_pulses = {"coupler_pulse": new_coupler_pulse}
        if qubit_driven:
            new_control_pulse = dict(old_control_pulse)
            new_control_pulse["amplitude"] = fit_results[edge_signature]["amplitude"]
            new_pulses["control_pulse"] = new_control_pulse
        else:
            new_coupler_pulse["amplitude"] = fit_results[edge_signature]["amplitude"]
        edge_parameters["new_parameter_values"][edge_key] = new_pulses

    return edge_parameters
