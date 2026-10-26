# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


"""This module defines the analysis for a JAZZ sequence using a CZ gate.

The experiment is defined in `laboneq_applications.contrib.experiments`.

In this analysis, we first interpret the raw data into qubit population using
principal-component analysis, or rotation and projection onto the measured
calibration states. Finally, we plot the data and the fit.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import uncertainties as unc
from laboneq import workflow
from laboneq.dsl.quantum import QPU
from laboneq.dsl.quantum.qpu_topology import TopologyEdge

from laboneq_applications.analysis.calibration_traces_rotation import (
    calculate_qubit_population_2d,
)
from laboneq_applications.analysis.fitting_helpers import (
    cosine_oscillatory_fit,
    get_pi_pi2_xvalues_on_cos,
)
from laboneq_applications.analysis.options import (
    ExtractEdgeParametersOptions,
    PlotPopulationOptions,
    TuneUpAnalysisWorkflowOptions,
)
from laboneq_applications.analysis.plotting_helpers import (
    plot_data_2d,
    plot_raw_complex_data_2d,
    timestamped_title,
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
    import lmfit
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
    """The JAZZ analysis workflow.

    The workflow consists of the following steps:

    - [extract_edges_from_qubit_pairs][]
    - [validate_flux_driven_edges][]
    - [extract_nodes_from_edges][]
    - [calculate_qubit_population][]
    - [fit_data][]
    - [extract_edge_parameters][]
    - [plot_fitted_frequencies][]
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

    # fit data for each single duration
    fit_results = fit_data(qubit_pairs, processed_data_dict)

    edges_parameters = extract_edge_parameters(edges, processed_data_dict, fit_results)

    with workflow.if_(options.do_plotting):
        plot_fitted_frequencies(processed_data_dict, fit_results)
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
) -> dict[str, ArrayLike[lmfit.model.ModelResult]]:
    """Fit a cosine oscillatory decay model to the qubit e-state population.

    If this fit fails an exponential-decay model fit will be performed instead under
    the assumption that the residual zz coupling is near zero.

    Arguments:
        qubit_pairs:
            The (control, target) qubit-UID pairs on which to run the task.
            The control UIDs must exist in `processed_data_dict`. Note that
            the task takes the qubit-pair UIDs rather than the topology
            edges, so that its input remains serializable by the logbook
            stores.
        processed_data_dict:
            The processed data dictionary containing the qubit population to be fitted
            and the sweep points of the experiment.
        options:
            The options for building the workflow as an instance of
            [FitData2QOptions][]. See the docstrings of this class
            for more details.

    Returns:
        Dictionary with qubit UIDs as keys and the fit results for each qubit as values.
    """
    opts = FitData2QOptions() if options is None else options

    fit_results = {}
    if not opts.do_fitting:
        return fit_results

    for q_control, q_target in qubit_pairs:
        amplitudes = processed_data_dict[q_control]["sweep_points_1d"]
        durations = processed_data_dict[q_control]["sweep_points_2d"]
        edge_signature = f"cz:{q_control}:{q_target}"
        fit_per_durations = {}
        for index, duration in enumerate(durations):
            # perform one oscillatiory fit per length of cz
            swpts_fit = amplitudes
            data_to_fit = processed_data_dict[q_control]["population"][index]

            try:
                fit_res = cosine_oscillatory_fit(
                    swpts_fit,
                    data_to_fit,
                    param_hints=opts.fit_parameters_hints,
                )
                # key by the duration index as a string so that the fit
                # results remain serializable by the logbook stores
                fit_per_durations[str(index)] = fit_res
            except ValueError as err:
                workflow.comment(
                    f"Fit failed for {edge_signature} at duration {duration}: {err}."
                )
        # add list of fits per edge
        fit_results[edge_signature] = fit_per_durations

    return fit_results


@workflow.task
def plot_population(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    options: PlotPopulationOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Create the JAZZ population plots.

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
            label_x_values="x90-Pulse Separation, $\\tau$ ($\\mu$s)",
            label_y_values="Flux Amplitude, $V_{cz}$ (V)",
            label_z_values="Principal Component (a.u)"
            if (num_cal_traces == 0 or opts.do_pca)
            else f"$|{opts.cal_states[-1]}\\rangle$-State Population",
            scaling_x_values=1e6,
            figure=fig,
            plot_title=f"Cz population {q.uid}",
            axis=axs,
            close_figures=opts.close_figures,
        )

        if opts.save_figures:
            workflow.save_artifact(f"CZ_population_{q.uid}", fig)

        figures[q.uid] = fig

    return figures


@workflow.task(save=False)
def extract_edge_parameters(
    edges: list[TopologyEdge],
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    fit_results: dict[str, dict[float, lmfit.model.ModelResult]],
    options: ExtractEdgeParametersOptions | None = None,
) -> dict[str, dict[str, dict[str, int | float | unc.core.Variable | None]]]:
    """Extract the edge parameters from the fit results.

    The pi-pulse amplitude fitted at the target duration becomes the new gate
    calibration. The duration goes into `coupler_pulse`, because the gate takes
    its length from there, and the amplitude into the pulse that drives the gate.

    Arguments:
        edges: The array of topology edges.
        processed_data_dict: The processed data dictionary.
        fit_results: The fit-results dictionary returned by `fit_data`.
        options:
            The options for extracting the qubit parameters.
            See [ExtractEdgeParameterOptions][] for accepted options.

    Returns:
        Dictionary with extracted tunable coupler parameters and the previous values
            for those tunable coupler parameters. The dictionary has the following form:

            ```python
            {
                "new_parameter_values": {
                    c.uid: {
                        tc_param_name: tc_param_value
                    },
                }
                "old_parameter_values": {
                    c.uid: {
                        tc_param_name: tc_param_value
                    },
                }
            }
            ```

            If the `do_fitting` option is False, the `new_parameter_values` are not
            extracted and the function only returns the `old_parameter_values`. If a
            qubit UID is not found in `fit_results`, the `new_parameter_values` entry
            for that qubit is left empty.
    """
    opts = ExtractEdgeParametersOptions() if options is None else options
    edge_parameters = {
        "old_parameter_values": {
            f"{e.tag}:{e.source_node.uid}:{e.target_node.uid}": {} for e in edges
        },
        "new_parameter_values": {
            f"{e.tag}:{e.source_node.uid}:{e.target_node.uid}": {} for e in edges
        },
    }

    for e in edges:
        # Extract parameters
        amplitudes = processed_data_dict[e.source_node.uid]["sweep_points_1d"]
        durations = processed_data_dict[e.source_node.uid]["sweep_points_2d"]
        # extract target duration for the fit, by default the middle duration
        target_index = len(durations) // 2
        target_duration = durations[target_index]
        # extract edge key and signature
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

        if not (opts.do_fitting and edge_signature in fit_results):
            continue

        # Extract and store the new pi and pi-half pulse amplitude values
        fit_res = fit_results[edge_signature].get(str(target_index))
        if fit_res is None:
            workflow.comment(
                f"No fit result at the target duration for {edge_signature}, "
                "skipping new parameter extraction."
            )
            continue
        swpts_fit = amplitudes
        freq_fit = unc.ufloat(
            fit_res.params["frequency"].value,
            fit_res.params["frequency"].stderr or np.nan,
        )
        phase_fit = unc.ufloat(
            fit_res.params["phase"].value,
            fit_res.params["phase"].stderr or np.nan,
        )
        (
            pi_amps_top,
            pi_amps_bottom,
            pi2_amps_rise,
            pi2_amps_fall,
        ) = get_pi_pi2_xvalues_on_cos(swpts_fit, freq_fit, phase_fit)

        # if pca is done, it can happen that the pi-pulse amplitude
        # is in pi_amps_bottom and the pi/2-pulse amplitude in pi2_amps_fall
        pi_amps = np.sort(np.concatenate([pi_amps_top, pi_amps_bottom]))
        pi2_amps = np.sort(np.concatenate([pi2_amps_rise, pi2_amps_fall]))
        try:
            pi2_amp = pi2_amps[0]
            pi_amp = pi_amps[pi_amps > pi2_amp][0]
        except IndexError:
            workflow.log(
                logging.ERROR,
                "Could not extract cz pulse amplitudes for %s.",
                edge_signature,
            )
            continue

        # the gate reads its length from the coupler pulse and its amplitude from
        # the pulse that drives it
        new_coupler_pulse = dict(old_coupler_pulse)
        new_coupler_pulse["length"] = target_duration
        new_pulses = {"coupler_pulse": new_coupler_pulse}
        if qubit_driven:
            new_control_pulse = dict(old_control_pulse)
            new_control_pulse["amplitude"] = pi_amp
            new_pulses["control_pulse"] = new_control_pulse
        else:
            new_coupler_pulse["amplitude"] = pi_amp
        edge_parameters["new_parameter_values"][edge_key] = new_pulses

    return edge_parameters


@workflow.task
def plot_fitted_frequencies(
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    fit_results: dict[str, ArrayLike[lmfit.model.ModelResult]],
    frequency_scaling_factor: float = 1e-3,
    options: PlotPopulationOptions | None = None,
) -> dict:
    """Plot the fitted frequencies per duration of the pulse.

    Arguments:
        processed_data_dict: The processed data dictionary.
        fit_results: The fit-results dictionary returned by `fit_data`.
        frequency_scaling_factor: The multiplication factor for better
            visualization in the plot.
        options:
            The options for extracting the qubit parameters.
            See [PlotPopulationOptions][] for accepted options.
    """
    opts = PlotPopulationOptions() if options is None else options
    figures = {}
    for edge_signature, result_dict in fit_results.items():
        q0 = edge_signature.split(":")[1]
        durations = processed_data_dict[q0]["sweep_points_2d"]
        durations_to_plot = []
        fitted_frequencies_to_plot = []
        fitted_decay_cnsts = []

        for index, length in enumerate(durations):
            result = result_dict.get(str(index))
            if result is not None:
                try:
                    fitted_frequencies_to_plot.append(
                        result.best_values["frequency"] * frequency_scaling_factor
                    )
                    durations_to_plot.append(length)
                except KeyError:
                    fitted_decay_cnsts.append(result.best_values.get("tau", None))
        fig, ax = plt.subplots()
        ax.plot(
            durations_to_plot,
            fitted_frequencies_to_plot,
            "o",
        )
        ax.set_ylabel("Fitted frequency (kHz)")
        ax.set_xlabel("Length of CZ $L_{cz}$ (s)")
        ax.set_title(timestamped_title(f"CZ coupling, fitted frequency {q0}"))

        figures[q0] = fig

        if opts.save_figures:
            workflow.save_artifact(f"fitted_frequency_shifts{q0}", fig)

        if opts.close_figures:
            plt.close(fig)

    return figures
