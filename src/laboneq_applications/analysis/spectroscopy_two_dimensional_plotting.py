# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the plotting analysis workflow for 2D spectroscopy experiments.

spectroscopy experiment workflows such as

- laboneq_applications.experiments.resonator_spectroscopy_amplitude
- laboneq_applications.experiments.qubit_spectroscopy_amplitude

In this analysis, we have the option to plot the raw data and the signal magnitude
and phase.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq import workflow

from laboneq_applications.analysis.plotting_helpers import (
    plot_raw_complex_data_2d,
    plot_signal_magnitude_and_phase_2d,
)

if TYPE_CHECKING:
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults

    from laboneq_applications.typing import QuantumElements, QubitSweepPoints


@workflow.workflow_options
class Spectroscopy2DPlottingWorkflowOptions:
    """Option class for the spectroscopy 2d plotting analysis workflows.

    Attributes:
        do_raw_data_plotting:
            Whether to plot the raw data.
            Default: True.
        do_plotting_magnitude_phase:
            Whether to plot the magnitude and phase.
            Default: True.
    """

    do_raw_data_plotting: bool = workflow.option_field(
        True, description="Whether to plot the raw data."
    )
    do_plotting_magnitude_phase: bool = workflow.option_field(
        True, description="Whether to plot the magnitude and phase."
    )


@workflow.workflow(name="analysis_workflow")
def analysis_workflow(
    result: RunExperimentResults,
    qubits: QuantumElements,
    sweep_points_1d: QubitSweepPoints,
    sweep_points_2d: QubitSweepPoints,
    label_sweep_points_1d: str,
    label_sweep_points_2d: str,
    scaling_sweep_points_2d: float = 1.0,
    options: Spectroscopy2DPlottingWorkflowOptions | None = None,
) -> None:
    """The analysis Workflow for plotting spectroscopy 2D data.

    The workflow consists of the following steps:

    - [plot_raw_complex_data_2d]()
    - [plot_signal_magnitude_and_phase_2d]()

    Arguments:
        result:
            The experiment results returned by the run_experiment task.
        qubits:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the result.
        sweep_points_1d:
            The sweep points corresponding to the innermost sweep.
            If `qubits` is a single qubit, `sweep_points_1d` must be a list of
            numbers or an array. Otherwise, it must be a list of lists of numbers or
            arrays.
        sweep_points_2d:
            The sweep points corresponding to the outermost sweep.
            If `qubits` is a single qubit, `sweep_points_2d` must be a list of
            numbers or an array. Otherwise, it must be a list of lists of numbers or
            arrays.
        label_sweep_points_1d:
            The label that will appear on the axis of the 1D sweep points.
            Passed to the `label_sweep_points_1d` parameter of
            `plot_raw_complex_data_2d`.
        label_sweep_points_2d:
            The label that will appear on the axis of the 2D sweep points.
            Passed to the `label_sweep_points_2d` parameter of
            `plot_raw_complex_data_2d`.
        scaling_sweep_points_2d:
            The scaling factor of the 2D sweep points.
            Passed to the `scaling_sweep_points_2d` parameter of
            `plot_raw_complex_data_2d`.
            Default: 1.0.
        options:
            The options for building the workflow, passed as an instance of
            [Spectroscopy2DPlottingWorkflowOptions]. See the docstring of this class
            for more details.

    Returns:
        WorkflowBuilder:
            The builder for the analysis workflow.

    Example:
        ```python
        result = analysis_workflow(
            results=results
            qubits=[q0, q1],
            frequencies=[
                np.linspace(6.0, 6.3, 301),
                np.linspace(5.8, 6.1, 301),
            ],
            options=analysis_workflow.options(),
        ).run()
        ```
    """
    with workflow.if_(options.do_raw_data_plotting):
        plot_raw_complex_data_2d(
            qubits=qubits,
            result=result,
            sweep_points_1d=sweep_points_1d,
            sweep_points_2d=sweep_points_2d,
            label_sweep_points_1d=label_sweep_points_1d,
            label_sweep_points_2d=label_sweep_points_2d,
            scaling_sweep_points_1d=1e-9,
            scaling_sweep_points_2d=scaling_sweep_points_2d,
        )
    with workflow.if_(options.do_plotting_magnitude_phase):
        plot_signal_magnitude_and_phase_2d(
            qubits=qubits,
            result=result,
            sweep_points_1d=sweep_points_1d,
            sweep_points_2d=sweep_points_2d,
            label_sweep_points_1d=label_sweep_points_1d,
            label_sweep_points_2d=label_sweep_points_2d,
            scaling_sweep_points_1d=1e-9,
            scaling_sweep_points_2d=scaling_sweep_points_2d,
        )
