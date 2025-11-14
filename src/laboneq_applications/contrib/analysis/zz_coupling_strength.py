# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


"""This module defines the analysis for a residual ZZ coupling strength experiment.

The experiment is defined in `laboneq_applications.contrib.experiments`.

In this analysis, we first interpret the raw data into qubit population using
principle-component analysis or rotation and projection on the measured calibration
states. Finally, we plot the data and the fit.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import uncertainties as unc
from laboneq import workflow
from laboneq.dsl.quantum import QPU
from laboneq.dsl.quantum.qpu_topology import TopologyEdge

from laboneq_applications.analysis.calibration_traces_rotation import (
    calculate_qubit_population_2d,
)
from laboneq_applications.analysis.fitting_helpers import (
    cosine_oscillatory_decay_fit,
    exponential_decay_fit,
)
from laboneq_applications.analysis.options import (
    ExtractEdgeParametersOptions,
    FitDataOptions,
    PlotPopulationOptions,
    TuneUpAnalysisWorkflowOptions,
)
from laboneq_applications.analysis.plotting_helpers import (
    plot_data_2d,
    plot_raw_complex_data_2d,
    timestamped_title,
)
from laboneq_applications.core.validation import (
    validate_and_convert_qubits_sweeps,
    validate_and_extract_edges_from_qubit_pairs,
)
from laboneq_applications.qpu_types.tunable_coupler import TunableCoupler
from laboneq_applications.tasks import extract_nodes_from_edges

if TYPE_CHECKING:
    import lmfit
    import matplotlib as mpl
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults
    from numpy.typing import ArrayLike

    from laboneq_applications.typing import (
        QuantumElements,
        QubitSweepPoints,
    )


@workflow.task_options(base_class=FitDataOptions)
class FitDataZZCouplingStrengthOptions:
    """Options for the `fit_data` task of the ZZ coupling strength analysis.

    See [FitDataOptions] for additional accepted options.

    Attributes:
        do_pca:
            Whether to perform principal component analysis on the raw data independent
            of whether there were calibration traces in the experiment.
            Default: `False`.
    """

    transition: Literal["ge", "ef"] = workflow.option_field(
        "ge",
        description="Transition to perform the experiment on. May be any"
        " transition supported by the quantum operations.",
    )
    do_pca: bool = workflow.option_field(
        False,
        description="Whether to perform principal component analysis on the raw data"
        " independent of whether there were calibration traces in the experiment.",
    )
    use_cal_traces: bool = workflow.option_field(
        True, description="Whether to include calibration traces in the experiment."
    )


@workflow.workflow
def analysis_workflow(
    result: RunExperimentResults,
    qpu: QPU,
    qubit_pairs: ArrayLike[ArrayLike[str, str]],
    biases: QubitSweepPoints,
    delays: QubitSweepPoints,
    options: TuneUpAnalysisWorkflowOptions | None = None,
) -> None:
    """The ZZ coupling strength analysis workflow.

    The workflow consists of the following steps:
    - [validate_and_extract_edges_from_qubit_pairs]()
    - [extract_nodes_from_edges]()
    - [calculate_qubit_population]()
    - [fit_data]()
    - [extract_edge_parameters]()
    - [plot_fitted_frequencies]()
    - [plot_raw_complex_data_2d]()
    - [plot_population2d]()

    Arguments:
        result:
            The experiment results returned by the `run_experiment` task.
        qpu:
            The quantum processing unit.
        qubit_pairs:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the result.
        biases:
            The coupler DC bias values swept over during the experiment.
        delays:
            The time between the first pi/2 pulse and the pi pulse and the time between
            the pi pulse and the second pi/2 pulse which was swept over in the
            experiment.
        options:
            The options for building the workflow, passed as an instance of
                [TuneUpAnalysisWorkflowOptions].
            In addition to options from [WorkflowOptions], the following
            custom options are supported: `do_fitting`, `do_plotting`, and the options
            of the [TuneupAnalysisOptions] class. See the docstring of
            [TuneUpAnalysisWorkflowOptions] for more details.

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
    edges = validate_and_extract_edges_from_qubit_pairs(
        qpu, "coupler", qubit_pairs, element_class=TunableCoupler
    )

    qubits_source = extract_nodes_from_edges(edges, "source")
    qubits_target = extract_nodes_from_edges(edges, "target")

    processed_data_dict = calculate_qubit_population_2d(
        qubits=qubits_source,
        result=result,
        sweep_points_1d=delays,
        sweep_points_2d=biases,
    )
    processed_data_dict_target = calculate_qubit_population_2d(
        qubits=qubits_target,
        result=result,
        sweep_points_1d=delays,
        sweep_points_2d=biases,
    )

    fit_results = fit_data(qubits_source, processed_data_dict)

    edges_parameters = extract_edge_parameters(edges, processed_data_dict, fit_results)

    with workflow.if_(options.do_plotting):
        plot_fitted_frequencies(processed_data_dict, fit_results)
        with workflow.if_(options.do_raw_data_plotting):
            plot_raw_complex_data_2d(
                qubits_source,
                result,
                sweep_points_1d=delays,
                sweep_points_2d=biases,
                label_sweep_points_2d="Coupler Bias, $V_{DC}$ (V)",
                label_sweep_points_1d="x90-Pulse Separation, $\\tau$ ($\\mu$s)",
                scaling_sweep_points_1d=1e6,
            )
            plot_raw_complex_data_2d(
                qubits_target,
                result,
                sweep_points_1d=delays,
                sweep_points_2d=biases,
                label_sweep_points_2d="Coupler Bias, $V_{DC}$ (V)",
                label_sweep_points_1d="x90-Pulse Separation, $\\tau$ ($\\mu$s)",
                scaling_sweep_points_1d=1e6,
            )
        with workflow.if_(options.do_qubit_population_plotting):
            plot_population(qubits_source, processed_data_dict)
            plot_population(qubits_target, processed_data_dict_target)

    workflow.return_(edges_parameters)


@workflow.task
def fit_data(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    options: FitDataZZCouplingStrengthOptions | None = None,
) -> dict[str, ArrayLike[lmfit.model.ModelResult]]:
    """Fit a cosine oscillatory decay model to the qubit e-state population.

    If this fit fails an exponential-decay model fit will be performed instead under
    the assumption that the residual zz coupling is near zero.

    Arguments:
        qubits:
            The qubits on which to run the task. The UIDs of these qubits must exist in
            `processed_data_dict`.
        processed_data_dict:
            The processed data dictionary containing the qubit population to be fitted
            and the sweep points of the experiment.
        options:
            The options for building the workflow as an instance of
            [FitDataZZCouplingsOptions]. See the docstrings of this class
            for more details.

    Returns:
        Dictionary with qubit UIDs as keys and the fit results for each qubit as values.
    """
    opts = FitDataZZCouplingStrengthOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)

    fit_results = {}
    if not opts.do_fitting:
        return None

    for q in qubits:
        sweep_points_1d = processed_data_dict[q.uid]["sweep_points_1d"]
        sweep_points_2d = processed_data_dict[q.uid]["sweep_points_2d"]
        fit_results_list = []
        for i in range(len(sweep_points_2d)):
            echo_pulse_length = (
                q.parameters.ef_drive_length
                if "f" in opts.transition
                else q.parameters.ge_drive_length
            )
            swpts_fit = sweep_points_1d + echo_pulse_length
            data_to_fit = processed_data_dict[q.uid]["population"][i]

            param_hints = {
                "amplitude": {"value": 0.5},
                "offset": {
                    "value": 0.5,
                    "vary": opts.do_pca,
                },  # or not opts.use_cal_traces},
            }
            param_hints_user = opts.fit_parameters_hints
            if param_hints_user is None:
                param_hints_user = {}
            param_hints.update(param_hints_user)
            try:
                fit_res = cosine_oscillatory_decay_fit(
                    swpts_fit,
                    data_to_fit,
                    param_hints=param_hints,
                )
                fit_results_list.append(fit_res)
            except ValueError as err:
                workflow.log(
                    logging.ERROR,
                    "Oscillatory decay fit failed for %s at bias value %s: %s.",
                    q.uid,
                    processed_data_dict[q.uid]["sweep_points_2d"][i],
                    err,
                )
                try:
                    fit_res = exponential_decay_fit(
                        swpts_fit,
                        data_to_fit,
                        param_hints=param_hints,
                    )
                except ValueError as err:
                    workflow.log(
                        logging.ERROR,
                        "Exponential decay fit also failed for %s at bias "
                        "value %s: %s.",
                        q.uid,
                        processed_data_dict[q.uid]["sweep_points_2d"][i],
                        err,
                    )
                finally:
                    fit_results_list.append(fit_res)

        fit_results[q.uid] = tuple(fit_results_list)

    return fit_results


@workflow.task
def plot_population(
    qubits: QuantumElements,
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    options: PlotPopulationOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Create the Hahn echo plots.

    Arguments:
        qubits:
            The qubits on which to run the task. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in
            `processed_data_dict`, `fit_results` and `qubit_parameters`.
        processed_data_dict:
            The processed data dictionary returned by `process_raw_data`.
        options:
            The options for processing the raw data.
            See [PlotPopulationOptions], ?[TuneupExperimentOptions]? and
            [BaseExperimentOptions] for accepted options.
            Overwrites the options from [PlotPopulationOptions],
            [TuneupExperimentOptions] and [BaseExperimentOptions].

    Returns:
        Dictionary with qubit UIDs as keys and the figures for each qubit as values.

        If a qubit UID is not found in `fit_results`, the fit and the textbox with the
        extracted qubit parameters are not plotted.
    """
    opts = PlotPopulationOptions() if options is None else options
    qubits = validate_and_convert_qubits_sweeps(qubits)

    figures = {}
    for q in qubits:
        num_cal_traces = processed_data_dict[q.uid]["num_cal_traces"]

        echo_pulse_length = (
            q.parameters.ef_drive_length
            if "f" in opts.transition
            else q.parameters.ge_drive_length
        )

        sweep_points_1d = (
            processed_data_dict[q.uid]["sweep_points_1d"] + echo_pulse_length
        )
        sweep_points_2d = processed_data_dict[q.uid]["sweep_points_2d"]
        data = processed_data_dict[q.uid][
            "population" if opts.do_rotation else "data_raw"
        ]

        fig, axs = plt.subplots()
        fig, axs = plot_data_2d(
            x_values=sweep_points_1d,
            y_values=sweep_points_2d,
            z_values=data,
            label_x_values="x90-Pulse Separation, $\\tau$ ($\\mu$s)",
            label_y_values="Coupler Bias, $V_{DC}$ (V)",
            label_z_values="Principal Component (a.u)"
            if (num_cal_traces == 0 or opts.do_pca)
            else f"$|{opts.cal_states[-1]}\\rangle$-State Population",
            scaling_x_values=1e6,
            figure=fig,
            plot_title=f"ZZ coupling strength {q.uid}",
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
    fit_results: dict[str, lmfit.model.ModelResult],
    options: ExtractEdgeParametersOptions | None = None,
) -> dict[str, dict[str, dict[str, int | float | unc.core.Variable | None]]]:
    """Extract the edge parameters from the fit results.

    Arguments:
        edges: The array of topology edges.
        processed_data_dict: The processed data dictionary.
        fit_results: The fit-results dictionary returned by `fit_data`.
        options:
            The options for extracting the qubit parameters.
            See [ExtractEdgeParameterOptions] for accepted options.

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
        extracted and the function only returns the `old_parameter_values`.
        If a qubit UID is not found in `fit_results`, the `new_parameter_values` entry
        for that qubit is left empty.
    """
    opts = ExtractEdgeParametersOptions() if options is None else options
    couplers: ArrayLike[TunableCoupler] = [e.quantum_element for e in edges]
    qubits: QuantumElements = [e.source_node for e in edges]
    edge_parameters = {
        "old_parameter_values": {c.uid: {} for c in couplers},
        "new_parameter_values": {c.uid: {} for c in couplers},
    }

    for q, c in zip(qubits, couplers):
        # Extract biases
        biases = processed_data_dict[q.uid]["sweep_points_2d"]

        # Store the previous coupler flux_voltage_offset
        old_flux_offset_voltage = c.parameters.flux_offset_voltage
        edge_parameters["old_parameter_values"][c.uid] = {
            "flux_voltage_offset": old_flux_offset_voltage
        }

        if opts.do_fitting and q.uid in fit_results:
            # Extract and store the minimal frequency fitted (smallest coupling)
            fits_array = fit_results[q.uid]
            best_fit_index = min(
                range(len(fits_array)),
                key=lambda i: fits_array[i].params["frequency"].value,
            )

            best_flux_voltage_offset = biases[best_fit_index]

            edge_parameters["new_parameter_values"][c.uid] = {
                "flux_voltage_offset": best_flux_voltage_offset,
            }

    return edge_parameters


@workflow.task
def plot_fitted_frequencies(
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    fit_results: dict[str, ArrayLike[lmfit.model.ModelResult]],
    frequency_scaling_factor: float = 1e-3,
    options: PlotPopulationOptions | None = None,
) -> dict:
    """Plot the fitted frequencies.

    Arguments:
        processed_data_dict: The processed data dictionary.
        fit_results: The fit-results dictionary returned by `fit_data`.
        frequency_scaling_factor: The multiplication factor for better
            visualization in the plot.
        options:
            The options for extracting the qubit parameters.
            See [PlotPopulationOptions] for accepted options.
    """
    opts = PlotPopulationOptions() if options is None else options
    figures = {}
    for quid, results in fit_results.items():
        biases = processed_data_dict[quid]["sweep_points_2d"]
        biases_to_plot = []
        fitted_frequencies_to_plot = []
        fitted_decay_cnsts = []

        for result, bias in zip(results, biases):
            if result is not None:
                try:
                    fitted_frequencies_to_plot.append(
                        result.best_values["frequency"] * frequency_scaling_factor
                    )
                    biases_to_plot.append(bias)
                except KeyError:
                    fitted_decay_cnsts.append(result.best_values.get("tau", None))
        fig, ax = plt.subplots()
        ax.plot(
            biases_to_plot,
            fitted_frequencies_to_plot,
            "o",
        )
        ax.set_ylabel("Fitted frequency (kHz)")
        ax.set_xlabel("Coupler DC bias $V_{DC}$ (V)")
        ax.set_title(timestamped_title(f"ZZ coupling, fitted frequnecy {quid}"))

        figures[quid] = fig

        if opts.save_figures:
            workflow.save_artifact(f"fitted_frequency_shifts{quid}", fig)

        if opts.close_figures:
            plt.close(fig)

    return figures
