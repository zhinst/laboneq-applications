# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Experiment task and workflow options."""

from __future__ import annotations

from typing import Literal, TypeVar

from laboneq.workflow import (
    option_field,
    task_options,
    workflow_options,
)

T = TypeVar("T")


@task_options
class DoFittingOption:
    """The `do_fitting` option.

    Attributes:
        do_fitting:
            Whether to perform the fit.
            Default: `True`.
    """

    do_fitting: bool = option_field(True, description="Whether to perform the fit.")


@task_options
class FitDataOptions:
    """Base options for data fitting tasks.

    Attributes:
        do_rotation:
            Whether to rotate the raw data based on calibration traces or principal
            component analysis.
            Default: `True`.
        do_fitting:
            Whether to perform the fit.
            Default: `True`.
        fit_parameters_hints:
            Parameters hints accepted by lmfit
            Default: None.
    """

    do_rotation: bool = option_field(
        True,
        description="Whether to rotate the raw data based on calibration traces or "
        "principal component analysis.",
    )
    do_fitting: bool = option_field(True, description="Whether to perform the fit.")
    fit_parameters_hints: dict | None = option_field(
        None, description="Parameters hints accepted by lmfit."
    )


@task_options
class ExtractQubitParametersTransitionOptions:
    """Base options for tasks that extract qubit parameters.

    Attributes:
        transition:
            Transition to perform the experiment on. May be any
            transition supported by the quantum operations.
            Default: `"ge"` (i.e. ground to first excited state).
        do_fitting:
            Whether to perform the fit.
            Default: `True`.
    """

    transition: Literal["ge", "ef"] = option_field(
        "ge",
        description="Transition to perform the experiment on. May be any"
        " transition supported by the quantum operations.",
    )
    do_fitting: bool = option_field(True, description="Whether to perform the fit.")


@task_options
class ExtractEdgeParametersOptions:
    """Base options for tasks that extract edge parameters.

    do_fitting:
        Whether to perform the fit.
        Default: `True`.
    """

    do_fitting: bool = option_field(True, description="Whether to perform the fit.")


@task_options
class BasePlottingOptions:
    """Base options for a plotting task.

    Attributes:
        save_figures:
            Whether to save the figures.
            Default: `True`.
        close_figures:
            Whether to close the figures.
            Default: `True`.

    """

    save_figures: bool = option_field(True, description="Whether to save the figures.")
    close_figures: bool = option_field(
        True, description="Whether to close the figures."
    )


@task_options
class PlotPopulationOptions(
    ExtractQubitParametersTransitionOptions, BasePlottingOptions
):
    """Options for the `plot_population` task.

    Attributes:
        do_rotation:
            Whether to rotate the raw data based on calibration traces or principal
            component analysis.
            Default: `True`.
        do_pca:
            Whether to perform principal component analysis on the raw data independent
            of whether there were calibration traces in the experiment.
            Default: `False`.
        cal_states:
            The states to prepare in the calibration traces. Can be any
            string or tuple made from combining the characters 'g', 'e', 'f'.
            Default: same as transition

    Additional attributes from `ExtractQubitParametersTransitionOptions`:
        transition:
            Transition to perform the experiment on. May be any
            transition supported by the quantum operations.
            Default: `"ge"` (i.e. ground to first excited state).
        do_fitting:
            Whether to perform the fit.
            Default: `True`.

    Additional attributes from `BasePlottingOptions`:
        save_figures:
            Whether to save the figures.
            Default: `True`.
        close_figures:
            Whether to close the figures.
            Default: `True`.
    """

    do_rotation: bool = option_field(
        True,
        description="Whether to rotate the raw data based on calibration traces or "
        "principal component analysis.",
    )
    do_pca: bool = option_field(
        False,
        description="Whether to perform principal component analysis on the raw data"
        " independent of whether there were calibration traces in the experiment.",
    )
    cal_states: str | tuple = option_field(
        "ge", description="The states to prepare in the calibration traces."
    )


@workflow_options
class TuneUpAnalysisWorkflowOptions:
    """Option class for tune-up analysis workflows.

    Attributes:
        do_fitting:
            Whether to perform the fit.
            Default: `True`.
        do_plotting:
            Whether to make plots.
            Default: 'True'.
        do_raw_data_plotting:
            Whether to plot the raw data.
            Default: 'True'.
        do_qubit_population_plotting:
            Whether to plot the qubit population.
            Default: 'True'.
    """

    do_fitting: bool = option_field(True, description="Whether to perform the fit.")
    do_plotting: bool = option_field(True, description="Whether to make plots.")
    do_raw_data_plotting: bool = option_field(
        True, description="Whether to plot the raw data."
    )
    do_qubit_population_plotting: bool = option_field(
        True, description="Whether to plot the qubit population."
    )
