# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the analysis for a signal propagation delay experiment.

The experiment is defined in laboneq_applications.experiments.

In this analysis, we extract the optimum integration delay defined by the maximum of
the integrated signal. Finally, we plot the data and mark the optimal delay.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from laboneq.workflow import (
    if_,
    save_artifact,
    task,
    task_options,
    workflow,
)

from laboneq_applications.analysis.options import (
    BasePlottingOptions,
    DoFittingOption,
    TuneUpAnalysisWorkflowOptions,
)

if TYPE_CHECKING:
    import matplotlib as mpl
    from laboneq.workflow.tasks.run_experiment import RunExperimentResults

    from laboneq_applications.typing import Qubit, QubitSweepPoints


@task_options
class PlotDataOption(DoFittingOption, BasePlottingOptions):
    """Option class for the `plot_data` task.

    Attributes from `DoFittingOption`:
        do_fitting:
            Whether to perform the fit.
            Default: `True`.

    Attributes from `BasePlottingOptions`:
        save_figures:
            Whether to save the figures.
            Default: `True`.
        close_figures:
            Whether to close the figures.
            Default: `True`.
    """


@workflow
def analysis_workflow(
    result: RunExperimentResults,
    qubit: Qubit,
    delays: QubitSweepPoints,
    options: TuneUpAnalysisWorkflowOptions | None = None,
) -> None:
    """The Amplitude Rabi analysis Workflow.

    The workflow consists of the following steps:

    - [extract_qubit_parameters]()
    - [plot_data]()

    Arguments:
        result:
            The experiment results returned by the run_experiment task.
        qubit:
            The qubits on which to run the analysis. May be either a single qubit or
            a list of qubits. The UIDs of these qubits must exist in the result.
        delays:
            The delays that were swept over in the signal propagation delay experiment.
            `delays` must be a list of numbers or an array.
        options:
            The options for building the workflow, passed as an instance of
            [TuneUpAnalysisWorkflowOptions]. See the docstring of this class for
            more details.

    Returns:
        WorkflowBuilder:
            The builder for the analysis workflow.

    Example:
        ```python
        options = TuneUpAnalysisWorkflowOptions()
        result = analysis_workflow(
            results=results
            qubits=q0,
            delays=np.linspace(0e-9, 200e-9, 21),
            ],
            options=options,
        ).run()
        ```
    """
    qubit_parameters = extract_qubit_parameters(qubit, result)
    with if_(options.do_plotting):
        with if_(options.do_raw_data_plotting):
            plot_data(
                qubit,
                result,
                qubit_parameters,
            )


@task
def extract_qubit_parameters(
    qubit: Qubit,
    result: RunExperimentResults,
    options: DoFittingOption | None = None,
) -> dict[str, dict[str, dict[str, int | float | None]]]:
    """Extract the optimal integration delay.

    The optimal integration delay is defined by the port_delay which results in a
    maximum of the integrated signal.

    Arguments:
        qubit:
            The qubit on which to run the analysis.
        result:
            The experiment results returned by the run_experiment task.
        options:
            The options for extracting the qubit parameters.
            See [DoFittingOption] for accepted options.

    Returns:
        dict with extracted qubit parameters and the previous values for those qubit
        parameters. The dictionary has the following form:
        ```python
        {
            "new_parameter_values": {
                q.uid: {
                    qb_param_name: qb_param_value
                },
            }
            "old_parameter_values": {
                q.uid: {
                    qb_param_name: qb_param_value
                },
            }
        }
        ```
        If the do_fitting option is False, the new_parameter_values are not extracted
        and the function only returns the old_parameter_values.
    """
    opts = DoFittingOption() if options is None else options
    q = qubit
    qubit_parameters = {
        "old_parameter_values": {q.uid: {}},
        "new_parameter_values": {q.uid: {}},
    }

    old_port_delay = q.parameters.readout_integration_delay
    qubit_parameters["old_parameter_values"][q.uid] = {
        "readout_integration_delay": old_port_delay,
    }
    if opts.do_fitting:
        iq_data = result[q.uid].result.data
        abs_data = np.abs(iq_data)
        swpts = result[q.uid].result.axis[0]
        good_delay = swpts[np.argmax(abs_data)]

        qubit_parameters["new_parameter_values"][q.uid] = {
            "readout_integration_delay": good_delay,
        }

    return qubit_parameters


@task
def plot_data(
    qubit: Qubit,
    result: RunExperimentResults,
    qubit_parameters: dict[
        str,
        dict[str, dict[str, int | float | None]],
    ]
    | None,
    options: PlotDataOption | None = None,
) -> dict[str, mpl.figure.Figure]:
    """Create the signal propagation delay plot.

    Arguments:
        qubit:
            The qubit on which to run the analysis.
            The UID of this qubit must exist in qubit_parameters.
        result: The experiment results returned by the run_experiment task.
        qubit_parameters: the qubit-parameters dictionary returned by
            extract_qubit_parameters
        options:
            The options for this task as an instance of [PlotDataOption].
            See the docstring of this class for more details.

    Returns:
        dict with qubit UID as key and the figure as values.
    """
    opts = PlotDataOption() if options is None else options
    figures = {}
    q = qubit
    swpts = result[q.uid].result.axis[0]
    iq_data = result[q.uid].result.data
    abs_data = np.abs(iq_data)

    fig, ax = plt.subplots()
    ax.set_title(f"Signal Propagation Delay {q.uid}")  # add timestamp here
    ax.set_xlabel("Port Delay, (ns)")
    ax.set_ylabel("Integrated Signal (a.u)")

    ax.plot(1e9 * swpts, abs_data, "o", zorder=2, label="data")
    if opts.do_fitting and len(qubit_parameters["new_parameter_values"][q.uid]) > 0:
        new_port_delay = qubit_parameters["new_parameter_values"][q.uid][
            "readout_integration_delay"
        ]
        # point at pi-pulse amplitude
        ax.plot(
            1e9 * new_port_delay,
            np.max(abs_data),
            "sk",
            zorder=3,
            markersize=plt.rcParams["lines.markersize"] + 1,
        )
        ylims = ax.get_ylim()
        ax.vlines(
            1e9 * new_port_delay,
            *ylims,
            linestyles="--",
            colors="gray",
            zorder=0,
            label="max.\nsignal",
        )
        ax.set_ylim(ylims)
        # textbox
        old_port_delay = qubit_parameters["old_parameter_values"][q.uid][
            "readout_integration_delay"
        ]
        textstr = f"Readout integration delay: {1e9*new_port_delay:.1f} ns"
        textstr += "\nOld readout integration delay: " + f"{1e9*old_port_delay:.1f} ns"
        ax.text(0, -0.15, textstr, ha="left", va="top", transform=ax.transAxes)
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            handlelength=1.5,
            frameon=False,
        )

        if opts.save_figures:
            save_artifact(f"Signal_Propagation_Delay{q.uid}", fig)

        if opts.close_figures:
            plt.close(fig)

        figures[q.uid] = fig

    return figures
