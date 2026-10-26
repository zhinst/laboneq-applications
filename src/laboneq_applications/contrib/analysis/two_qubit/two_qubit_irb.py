# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


"""Analysis for a two-qubit interleaved randomized benchmarking experiment.

The experiment is defined in
`laboneq_applications.contrib.experiments.two_qubit.two_qubit_irb`.

The single-shot readout results of the two qubits of each pair are sorted into
two-qubit basis-state counts, which are then handed back to Qiskit's built-in
analysis workflow for interleaved RB experiments to obtain the error per
Clifford and the error per interleaved gate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from laboneq.dsl.quantum import QPU
from laboneq.dsl.quantum.qpu_topology import TopologyEdge
from laboneq.simple import AveragingMode
from laboneq.workflow import (
    comment,
    if_,
    option_field,
    save_artifact,
    task,
    task_options,
    workflow,
    workflow_options,
)
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit_experiments.database_service.exceptions import (
    ExperimentEntryNotFound,
)
from qiskit_experiments.framework import ExperimentData
from qiskit_experiments.library import randomized_benchmarking

from laboneq_applications.analysis.options import PlotPopulationOptions
from laboneq_applications.core.validation import (
    validate_flux_driven_edges,
    validate_result,
)
from laboneq_applications.tasks import extract_edges_from_qubit_pairs

if TYPE_CHECKING:
    from typing import Literal

    import matplotlib as mpl
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults
    from numpy.typing import ArrayLike
    from qiskit_experiments.library.randomized_benchmarking import (
        InterleavedRB,
    )


@workflow_options
class TwoQubitIRBTuneUpAnalysisWorkflowOptions:
    """Options for the two-qubit interleaved RB analysis workflow."""

    do_plotting: bool = option_field(default=True)
    interleaved_gate: Literal["iswap", "cz", "cx"] = option_field(
        "cz",
        description="The gate to benchmark with interleaved randomized benchmarking.",
    )


@task_options
class ProcessRawDataOptions:
    """Options for the process_raw_data task."""

    count: int = option_field(default=1024, description="The number of repetitions.")
    averaging_mode: AveragingMode = option_field(
        default=AveragingMode.SINGLE_SHOT, converter=AveragingMode
    )


@task_options
class RunAnalysisOptions:
    """Options for the run_analysis task."""

    count: int = option_field(default=1024, description="The number of repetitions.")
    interleaved_gate: Literal["iswap", "cz", "cx"] = option_field(
        "cz",
        description="The gate to benchmark with interleaved randomized benchmarking.",
    )


@task_options(base_class=PlotPopulationOptions)
class PlotIRBOptions:
    """Options for the create_irb_sequences task.

    Attributes:
        interleaved_gate: The gate to benchmark with interleaved randomized benchmarking
        two_q_uid: Key for storing the two-qubit system info.
    """


@workflow(name="two qubit IRB analysis workflow")
def analysis_workflow(
    result: RunExperimentResults,
    qpu: QPU,
    qubit_pairs: list[list[str, str]],
    length_cliffords: list,
    variations: int,
    options: TwoQubitIRBTuneUpAnalysisWorkflowOptions | None = None,
) -> None:
    """The two-qubit interleaved randomized benchmarking analysis workflow.

    The workflow consists of the following steps:

    - [extract_edges_from_qubit_pairs][]
    - [validate_flux_driven_edges][]
    - [generate_qiskit_containers][]
    - [process_raw_data][]
    - [run_analysis][]
    - [plot_irb][]

    Arguments:
        result:
            The experiment results returned by the `run_experiment` task.
        qpu:
            The quantum processing unit.
        qubit_pairs:
            The qubit pairs the experiment was run on, passed by UID.
        length_cliffords:
            The Clifford-sequence lengths used in the experiment.
        variations:
            The number of random sequences per Clifford-sequence length.
        options:
            The options for building the workflow, passed as an instance of
            [TwoQubitIRBTuneUpAnalysisWorkflowOptions][]. See the docstring of
            that class for more details.

    Returns:
        WorkflowBuilder:
            The builder for the analysis workflow.

    """
    edges = extract_edges_from_qubit_pairs(
        qpu,
        qubit_pairs,
        tag=options.interleaved_gate,
    )
    validate_flux_driven_edges(edges)
    qiskit_containers = generate_qiskit_containers(length_cliffords, variations)
    processed_data_dict = process_raw_data(
        edges, result, length_cliffords, variations, qiskit_containers
    )

    processed_data_dict = run_analysis(edges, processed_data_dict)
    with if_(options.do_plotting):
        plot_irb(edges, processed_data_dict)


@task
def generate_qiskit_containers(
    length_cliffords: list, variations: int
) -> InterleavedRB:
    """Create the Qiskit interleaved RB experiment container."""
    # all qiskit backends support cx so we will use this as a dummy interleaved gate
    dummy_interleaved_gate = get_standard_gate_name_mapping()["cx"]
    return randomized_benchmarking.InterleavedRB(
        interleaved_element=dummy_interleaved_gate,
        physical_qubits=[0, 1],
        lengths=length_cliffords,
        num_samples=variations,
    )


@task
def process_raw_data(
    edges: list[TopologyEdge],
    result: RunExperimentResults,
    length_cliffords: list,
    variations: int,
    qiskit_containers: InterleavedRB,
    options: ProcessRawDataOptions | None = None,
) -> dict[str, dict[str, ArrayLike]]:
    """Sort and process the raw results.

    For each edge, the single-shot readout results of the two qubits are mapped
    onto the two-qubit basis states "00", "01", "10" and "11" and accumulated
    into per-circuit counts, in the form that Qiskit's randomized benchmarking
    analysis expects.

    Arguments:
        edges:
            The topology edges of the qubit pairs the experiment was run on.
        result:
            The experiment results returned by the `run_experiment` task.
        length_cliffords:
            The Clifford-sequence lengths used in the experiment.
        variations:
            The number of random sequences per Clifford-sequence length.
        qiskit_containers:
            The Qiskit `InterleavedRB` experiment that produced the sequences.
        options:
            The options for processing the raw data, passed as an instance of
            [ProcessRawDataOptions][]. Only data acquired with
            `AveragingMode.SINGLE_SHOT` can be processed.

    Returns:
        Dictionary with the edge signatures as keys and the Clifford lengths,
            the Qiskit container and the Qiskit data for each edge as values.
    """
    validate_result(result)
    opts = ProcessRawDataOptions() if options is None else options

    if opts.averaging_mode != AveragingMode.SINGLE_SHOT:
        msg = (
            "Only data acquired with "
            "options.averaging_mode(AveragingMode.SINGLE_SHOT) "
            "can be analyzed in this workflow"
        )
        raise ValueError(msg)

    processed_data_dict = {}
    # loop over number of edges
    for e in edges:
        edge_signature = f"{e.tag}:{e.source_node.uid}:{e.target_node.uid}"
        cliffords = np.concatenate([length_cliffords for _ in range(variations)])
        q0_raw_data = result.data[e.source_node.uid].result.data
        q1_raw_data = result.data[e.target_node.uid].result.data

        state_0 = 0.0 + 0.0j
        state_1 = 1.0 + 0.0j

        two_q_state_dict = {
            (state_0, state_0): "00",
            (state_1, state_0): "10",
            (state_0, state_1): "01",
            (state_1, state_1): "11",
        }

        circuit_data = []
        for job_num, circuit in enumerate(qiskit_containers.circuits()):
            counts = {}
            memory = []
            for shot in range(opts.count):
                two_q_state = two_q_state_dict[
                    (q0_raw_data[shot, job_num], q1_raw_data[shot, job_num])
                ]
                counts[two_q_state] = counts.get(two_q_state, 0) + 1
                memory.append(two_q_state)

            circuit_data.append(
                {
                    "counts": counts,
                    "memory": memory,
                    "metadata": circuit.metadata,
                    "shots": opts.count,
                }
            )

        qiskit_data = ExperimentData(experiment=qiskit_containers)
        qiskit_data.add_data(circuit_data)

        processed_data_dict[edge_signature] = {
            "cliffords": cliffords,
            "qiskit_containers": qiskit_containers,
            "qiskit_data": qiskit_data,
        }

    return processed_data_dict


@task
def run_analysis(
    edges: list[TopologyEdge],
    processed_data_dict: dict,
    options: RunAnalysisOptions | None = None,
) -> dict:
    """Run the Qiskit interleaved RB analysis on the processed data."""
    opts = RunAnalysisOptions() if options is None else options
    for e in edges:
        edge_signature = f"{e.tag}:{e.source_node.uid}:{e.target_node.uid}"
        two_q_data_dict = processed_data_dict[edge_signature]

        qiskit_containers = two_q_data_dict["qiskit_containers"]
        qiskit_data = two_q_data_dict["qiskit_data"]

        plotter = qiskit_containers.analysis.plotter
        plotter.set_figure_options(
            xlabel="Number of Cliffords",
            ylabel="P(0)",
            figure_title=(
                f"IRB | Interleaved: {opts.interleaved_gate} | Qubits: {edge_signature}"
            ),
        )

        qiskit_data = qiskit_containers.analysis.run(
            qiskit_data, replace_results=True
        ).block_for_results()
        two_q_data_dict.update({"qiskit_data": qiskit_data})
    return processed_data_dict


@task
def plot_irb(
    edges: list[TopologyEdge],
    processed_data_dict: dict[str, dict[str, ArrayLike]],
    options: PlotIRBOptions | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Create the IRB plot.

    Arguments:
        edges:
            The topology edges of the qubit pairs to plot.
        processed_data_dict:
            The processed data dictionary returned by run_analysis.
        options:
            The options class for this task as an instance of [PlotIRBOptions][].
            See the docstring of this class for accepted options.
    """
    opts = PlotIRBOptions() if options is None else options
    for e in edges:
        edge_signature = f"{e.tag}:{e.source_node.uid}:{e.target_node.uid}"
        qiskit_data = processed_data_dict[edge_signature].get("qiskit_data", None)
        try:
            fig = qiskit_data.figure(0)
        except ExperimentEntryNotFound:
            comment("Figures in Qiskit data were out of range")
            fig = None

        if opts.save_figures and fig is not None:
            save_artifact(
                "two_q_irb",
                fig.figure,
                options={"format": "png"},
            )
        if not opts.close_figures:
            # lazy import: IPython is not a declared dependency
            # of laboneq_applications
            from IPython.display import display  # noqa: PLC0415

            display(fig)
