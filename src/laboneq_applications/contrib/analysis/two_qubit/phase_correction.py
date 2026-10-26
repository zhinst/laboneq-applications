# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


"""This module defines the analysis for a CZ phase-correction experiment.

The experiment is defined in
`laboneq_applications.contrib.experiments.two_qubit.phase_correction`.

In this analysis, we first interpret the raw data into qubit population using
principal-component analysis, or rotation and projection onto the measured
calibration states. We then fit an oscillation to the phase sweep of each qubit
of the pair, and extract the single-qubit phase corrections that have to be
applied after the CZ gate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import uncertainties as unc
from laboneq import workflow
from laboneq.dsl.quantum import QPU
from laboneq.dsl.quantum.qpu_topology import TopologyEdge
from laboneq.simple import dsl

from laboneq_applications.analysis.calibration_traces_rotation import (
    CalculateQubitPopulationOptions,
    calculate_population_1d,
    extract_raw_data_dict,
)
from laboneq_applications.analysis.fitting_helpers import (
    cosine_oscillatory_fit,
)
from laboneq_applications.analysis.options import (
    ExtractEdgeParametersOptions,
    FitDataOptions,
    TuneUpAnalysisWorkflowOptions,
)
from laboneq_applications.analysis.plotting_helpers import (
    plot_raw_complex_data_1d,
)
from laboneq_applications.conversion import sweeps_per_quantum_element
from laboneq_applications.core.validation import (
    validate_flux_driven_edges,
    validate_result,
)
from laboneq_applications.tasks import (
    extract_edges_from_qubit_pairs,
    extract_nodes_from_edges,
)

if TYPE_CHECKING:
    import lmfit
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
    phases: QubitSweepPoints,
    options: TuneUpAnalysisWorkflowOptions | None = None,
) -> None:
    """The CZ phase-correction analysis workflow.

    The workflow consists of the following steps:

    - [extract_edges_from_qubit_pairs][]
    - [validate_flux_driven_edges][]
    - [extract_nodes_from_edges][]
    - [calculate_qubit_population][]
    - [fit_data][]
    - [extract_edge_parameters][]
    - [plot_raw_complex_data_1d][]
    - [plot_population][]

    Arguments:
        result:
            The experiment results returned by the `run_experiment` task.
        qpu:
            The quantum processing unit.
        qubit_pairs:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the result.
        phases:
            The phase shift values applied on the qubit that the phase is
            being measured on.
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

    processed_data_dict_control = calculate_qubit_population(
        qubits=qubits_source,
        result=result,
        sweep_points=phases,
        suffix="control",
    )
    processed_data_dict_target = calculate_qubit_population(
        qubits=qubits_target,
        result=result,
        sweep_points=phases,
        suffix="target",
    )

    fit_results = fit_data(
        qubit_pairs,
        processed_data_dict_control,
        processed_data_dict_target,
    )

    edges_parameters = extract_edge_parameters(
        edges,
        fit_results,
    )

    with workflow.if_(options.do_plotting):
        with workflow.if_(options.do_raw_data_plotting):
            plot_raw_complex_data_1d(
                qubits_source,
                result,
                sweep_points=phases,
                xlabel="Phase sweep, $theta$ (rad)",
                xscaling=1,
            )
            plot_raw_complex_data_1d(
                qubits_target,
                result,
                sweep_points=phases,
                xlabel="Phase sweep, $theta$ (rad)",
                xscaling=1,
            )

    workflow.return_(edges_parameters)


@workflow.task
def calculate_qubit_population(
    qubits: QuantumElements,
    result: RunExperimentResults,
    sweep_points: QubitSweepPoints,
    suffix: str = "",
    options: CalculateQubitPopulationOptions | None = None,
) -> dict[str, dict[str, ArrayLike]]:
    """Calculates the qubit population from the raw data.

    The data is processed in the following way:

    - If calibration traces were used in the experiment, the raw data is rotated
      based on the calibration traces. See
      [rotate_data_to_cal_trace_results][laboneq_applications.analysis.calibration_traces_rotation.rotate_data_to_cal_trace_results]
      for more details.
    - If no calibration traces were used in the experiment, or `do_pca = True` is
      passed in options, principal-component analysis is performed on the
      data. See
      [principal_component_analysis][laboneq_applications.analysis.calibration_traces_rotation.principal_component_analysis]
      for more details.

    Arguments:
        qubits:
            The qubits on which the amplitude-Rabi experiments was run. May be either
            a single qubit or a list of qubits.
        result: the result of the experiment, returned by the run_experiment task.
        sweep_points:
            The sweep points used in the experiment for each qubit. If `qubits` is a
            single qubit, `sweep_points` must be an array. Otherwise, it must be a list
            of arrays.
        suffix: to add to the dsl handle when accessing the result data.
        options:
            The options for building the workflow as an instance of
            [CalculateQubitPopulationOptions][]. See the docstrings of this class for more
            details.

    Returns:
        dict with qubit UIDs as keys and the dictionary of processed data for each qubit
            as values. See
            [calculate_population_1d][laboneq_applications.analysis.calibration_traces_rotation.calculate_population_1d]
            for what this dictionary looks like.

    Raises:
        TypeError:
            If result is not an instance of RunExperimentResults.
        ValueError:
            If the qubits and sweeps have incompatible shapes.
    """
    opts = CalculateQubitPopulationOptions() if options is None else options
    validate_result(result)
    processed_data_dict = {}
    for q, swpts in sweeps_per_quantum_element(qubits, sweep_points):
        raw_data = result[dsl.handles.result_handle(q.uid, suffix=suffix)].data
        if opts.use_cal_traces:
            calibration_traces = [
                result[dsl.handles.calibration_trace_handle(q.uid, cs)].data
                for cs in opts.cal_states
            ]
            do_pca = opts.do_pca
        else:
            calibration_traces = []
            do_pca = True

        if opts.do_rotation:
            data_dict = calculate_population_1d(
                raw_data,
                swpts,
                calibration_traces,
                do_pca=do_pca,
            )
        else:
            data_dict = extract_raw_data_dict(raw_data, swpts, calibration_traces)
        processed_data_dict[q.uid] = data_dict
    return processed_data_dict


@workflow.task
def fit_data(
    qubit_pairs: list[list[str]],
    processed_data_dict_control: dict[str, dict[str, ArrayLike]],
    processed_data_dict_target: dict[str, dict[str, ArrayLike]],
    options: FitDataOptions | None = None,
) -> dict[str, lmfit.model.ModelResult]:
    """Perform a fit of a cosine model to the data.

    Arguments:
        qubit_pairs:
            The (control, target) qubit-UID pairs on which to run the task.
            Note that the task takes the qubit-pair UIDs rather than the
            topology edges, so that its input remains serializable by the
            logbook stores.
        processed_data_dict_control:
            The processed data dictionary for the control qubits, returned by
            calculate_qubit_population.
        processed_data_dict_target:
            The processed data dictionary for the target qubits, returned by
            calculate_qubit_population.
        options:
            The options class for this task as an instance of [FitDataOptions][]. See
            the docstring of this class for accepted options.

    Returns:
        dict with qubit UIDs as keys and the fit results for each qubit as keys.
    """
    opts = FitDataOptions() if options is None else options
    fit_results = {}
    if not opts.do_fitting:
        return fit_results

    for q_control, q_target in qubit_pairs:
        swpts_fit = processed_data_dict_control[q_control]["sweep_points"]
        data_to_fit_control = processed_data_dict_control[q_control][
            "population" if opts.do_rotation else "data_raw"
        ]
        data_to_fit_target = processed_data_dict_target[q_target][
            "population" if opts.do_rotation else "data_raw"
        ]
        edge_signature = f"cz:{q_control}:{q_target}"
        try:
            fit_res_control = cosine_oscillatory_fit(
                swpts_fit,
                data_to_fit_control,
                param_hints=opts.fit_parameters_hints,
            )
            fit_res_target = cosine_oscillatory_fit(
                swpts_fit,
                data_to_fit_target,
                param_hints=opts.fit_parameters_hints,
            )
            fit_results[edge_signature] = {
                "fit_results_control": fit_res_control,
                "fit_results_target": fit_res_target,
            }
        except ValueError as err:
            workflow.comment(f"Fit failed for {edge_signature}: {err}.")
            fit_results[edge_signature] = {
                "fit_results_control": None,
                "fit_results_target": None,
            }

    return fit_results


@workflow.task(save=False)
def extract_edge_parameters(
    edges: list[TopologyEdge],
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
    edge_parameters = {
        "old_parameter_values": {
            f"{e.tag}:{e.source_node.uid}:{e.target_node.uid}": {} for e in edges
        },
        "new_parameter_values": {
            f"{e.tag}:{e.source_node.uid}:{e.target_node.uid}": {} for e in edges
        },
    }

    for e in edges:
        # extract edge key and signature
        edge_key = f"{e.tag}:{e.source_node.uid}:{e.target_node.uid}"
        edge_signature = edge_key

        # Store the previous phase-correction angles of the edge
        old_angles = {
            "control_angle": e.parameters.control_angle,
            "target_angle": e.parameters.target_angle,
        }
        edge_parameters["old_parameter_values"][edge_key] = old_angles
        # get the two fits; fit_data omits the edge entirely if fitting was not done
        fits = fit_results.get(edge_signature, {})
        fit_control = fits.get("fit_results_control")
        fit_result_target = fits.get("fit_results_target")
        if fit_control is None or fit_result_target is None:
            workflow.comment(
                f"No fit results for {edge_signature}, "
                f"skipping new parameter extraction."
            )
        else:
            # collect values for angle correction
            new_angles = {
                "control_angle": fit_control.params["phase"].value,
                "target_angle": fit_result_target.params["phase"].value,
            }
            edge_parameters["new_parameter_values"][edge_key] = new_angles

    return edge_parameters
