"""Experiment task and workflow options."""

from __future__ import annotations

from typing import Literal, TypeVar

from laboneq.simple import AcquisitionType, AveragingMode, RepetitionMode
from laboneq.workflow import TaskOptions, WorkflowOptions, option_field, options

T = TypeVar("T")


def _parse_acquisition_type(v: str | AcquisitionType) -> AcquisitionType:
    return AcquisitionType(v)


def _parse_averaging_mode(v: str | AveragingMode) -> AveragingMode:
    return AveragingMode(v)


def _parse_repetition_mode(v: str | RepetitionMode) -> RepetitionMode:
    return RepetitionMode(v)


@options
class BaseExperimentOptions(TaskOptions):
    """Base options for the experiment.

    Attributes:
        count:
            The number of repetitions.
            Default: A common choice in practice, 1024.
        averaging_mode:
            Averaging mode to use for the experiment.
            Default: `AveragingMode.CYCLIC`.
        acquisition_type:
            Acquisition type to use for the experiment.
            Default: `AcquisitionType.INTEGRATION`.
        repetition_mode:
            The repetition mode to use for the experiment.
            Default: `RepetitionMode.FASTEST`.
        repetition_time:
            The repetition time.
            Default: None.
        reset_oscillator_phase:
            Whether to reset the oscillator phase.
            Default: False.
    """

    count: int = option_field(default=1024, description="The number of repetitions.")
    acquisition_type: str | AcquisitionType = option_field(
        AcquisitionType.INTEGRATION,
        description="Acquisition type to use for the experiment.",
        converter=_parse_acquisition_type,
    )
    averaging_mode: str | AveragingMode = option_field(
        AveragingMode.CYCLIC,
        description="Averaging mode to use for the experiment.",
        converter=_parse_averaging_mode,
    )
    repetition_mode: str | RepetitionMode = option_field(
        RepetitionMode.FASTEST,
        description="The repetition mode to use for the experiment.",
        converter=_parse_repetition_mode,
    )
    repetition_time: float | None = option_field(
        None, description="The repetition time."
    )
    reset_oscillator_phase: bool = option_field(
        False, description="Whether to reset the oscillator phase."
    )


@options
class TuneupExperimentOptions(BaseExperimentOptions):
    """Base options for a tune-up experiment.

    Attributes:
        transition:
            Transition to perform the experiment on. May be any
            transition supported by the quantum operations.
            Default: `"ge"` (i.e. ground to first excited state).
        use_cal_traces:
            Whether to include calibration traces in the experiment.
            Default: `True`.
        cal_states:
            The states to prepare in the calibration traces. Can be any
            string or tuple made from combining the characters 'g', 'e', 'f'.
            Default: same as transition
    """

    transition: Literal["ge", "ef"] = option_field(
        "ge",
        description="Transition to perform the experiment on. May be any"
        " transition supported by the quantum operations.",
    )
    use_cal_traces: bool = option_field(
        True, description="Whether to include calibration traces in the experiment."
    )
    cal_states: str | tuple = option_field(
        "ge", description="The states to prepare in the calibration traces."
    )


# create additional options for spectroscopy
@options
class ResonatorSpectroscopyExperimentOptions(BaseExperimentOptions):
    """Base options for the resonator spectroscopy experiment.

    Additional attributes:
        use_cw:
            Perform a CW spectroscopy where no measure pulse is played.
            Default: False.
        spectroscopy_reset_delay:
            How long to wait after an acquisition in seconds.
            Default: 1e-6.
        acquisition_type:
            Acquisition type to use for the experiment.
            Default: `AcquisitionType.SPECTROSCOPY`.
    """

    use_cw: bool = option_field(
        False, description="Perform a CW spectroscopy where no measure pulse is played."
    )
    spectroscopy_reset_delay: float = option_field(
        1e-6, description="How long to wait after an acquisition in seconds."
    )
    acquisition_type: AcquisitionType = option_field(
        AcquisitionType.SPECTROSCOPY,
        description="Acquisition type to use for the experiment.",
    )


@options
class TuneupAnalysisOptions(TuneupExperimentOptions):
    """Base options for the analysis of a tune-up experiment.

    Attributes:
        do_rotation:
            Whether to rotate the raw data based on calibration traces or principal
            component analysis.
            Default: `True`.
        do_pca:
            Whether to perform principal component analysis on the raw data independent
            of whether there were calibration traces in the experiment.
            Default: `False`.
        do_fitting:
            Whether to perform the fit.
            Default: `True`.
        fit_parameters_hints:
            Parameters hints accepted by lmfit
            Default: None.
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
    do_fitting: bool = option_field(True, description="Whether to perform the fit.")
    fit_parameters_hints: dict | None = option_field(
        None, description="Parameters hints accepted by lmfit."
    )
    save_figures: bool = option_field(True, description="Whether to save the figures.")
    close_figures: bool = option_field(
        True, description="Whether to close the figures."
    )


@options
class TuneUpAnalysisWorkflowOptions(WorkflowOptions):
    """Option class for tune-up analysis workflows.

    Attributes:
        do_fitting:
            Whether to perform the fit.
            Default: `True`.
        do_plotting:
            Whether to make plots.
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


@options
class TuneUpWorkflowOptions(WorkflowOptions):
    """Option class for tune-up experiment workflows.

    Attributes:
        do_analysis (bool):
            Whether to run the analysis workflow.
            Default: True
        update (bool):
            Whether to update the setup based on the results from the analysis.
            Default: False
    """

    do_analysis: bool = option_field(
        True, description="Whether to run the analysis workflow."
    )
    update: bool = option_field(
        False,
        description="Whether to update the setup based on the "
        "results from the analysis.",
    )


@options
class QubitSpectroscopyExperimentOptions(BaseExperimentOptions):
    """Base options for the resonator spectroscopy experiment.

    Additional attributes:
        spectroscopy_reset_delay:
            How long to wait after an acquisition in seconds.
            Default: 1e-6.
    """

    spectroscopy_reset_delay: float = option_field(
        1e-6, description="How long to wait after an acquisition in seconds."
    )


@options
class QubitSpectroscopyAnalysisOptions(QubitSpectroscopyExperimentOptions):
    """Base options for the analysis of a qubit-spectroscopy experiment.

    Attributes:
        do_fitting:
            Whether to perform the fit.
            Default: `True`.
        fit_parameters_hints:
            Parameters hints accepted by lmfit
            Default: None.
        save_figures:
            Whether to save the figures.
            Default: `True`.
        close_figures:
            Whether to close the figures.
            Default: `True`.
    """

    do_fitting: bool = option_field(True, description="Whether to perform the fit.")
    fit_parameters_hints: dict | None = option_field(
        None, description="Parameters hints accepted by lmfit."
    )
    save_figures: bool = option_field(True, description="Whether to save the figures.")
    close_figures: bool = option_field(
        True, description="Whether to close the figures."
    )


@options
class QubitSpectroscopyAnalysisWorkflowOptions(WorkflowOptions):
    """Option class for qubit spectroscopy analysis workflows.

    Attributes:
        do_plotting:
            Whether to make plots.
            Default: 'True'.
        do_raw_data_plotting:
            Whether to plot the raw data.
            Default: True.
        do_plotting_qubit_spectroscopy:
            Whether to plot the final qubit spectroscopy plot.
            Default: True.
    """

    do_plotting: bool = option_field(True, description="Whether to make plots.")
    do_raw_data_plotting: bool = option_field(
        True, description="Whether to plot the raw data."
    )
    do_plotting_qubit_spectroscopy: bool = option_field(
        True, description="Whether to plot the final qubit spectroscopy plot."
    )
