"""Experiment task and workflow options."""

from __future__ import annotations

from typing import Annotated, Literal, TypeVar

from laboneq.simple import AcquisitionType, AveragingMode, RepetitionMode
from pydantic import (
    Field,
    field_validator,
    model_validator,
)

from laboneq_applications.workflow.options import TaskOptions, WorkflowOptions

NonNegativeInt = Annotated[int, Field(ge=0)]
T = TypeVar("T")


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

    count: NonNegativeInt = Field(
        default=1024, description="The number of repetitions."
    )
    acquisition_type: str | AcquisitionType = Field(
        AcquisitionType.INTEGRATION,
        description="Acquisition type to use for the experiment.",
    )
    averaging_mode: str | AveragingMode = Field(
        AveragingMode.CYCLIC, description="Averaging mode to use for the experiment."
    )
    repetition_mode: str | RepetitionMode = Field(
        RepetitionMode.FASTEST,
        description="The repetition mode to use for the experiment.",
    )
    repetition_time: float | None = Field(None, description="The repetition time.")
    reset_oscillator_phase: bool = Field(
        False, description="Whether to reset the oscillator phase."
    )

    @field_validator("acquisition_type", mode="after")
    @classmethod
    def _parse_acquisition_type(cls, v: str) -> AcquisitionType:
        # parse string to AcquisitionType
        return AcquisitionType(v)

    @field_validator("averaging_mode", mode="after")
    @classmethod
    def _parse_averaging_mode(cls, v: str) -> AveragingMode:
        # parse string to AveragingMode
        return AveragingMode(v)

    @field_validator("repetition_mode", mode="after")
    @classmethod
    def _parse_repetition_mode(cls, v: str) -> RepetitionMode:
        # parse string to RepetitionMode
        return RepetitionMode(v)


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

    transition: Literal["ge", "ef"] = Field(
        "ge",
        description="Transition to perform the experiment on. May be any"
        " transition supported by the quantum operations.",
    )
    use_cal_traces: bool = Field(
        True, description="Whether to include calibration traces in the experiment."
    )
    cal_states: str | tuple = Field(
        "ge", description="The states to prepare in the calibration traces."
    )

    @model_validator(mode="before")
    @classmethod
    def _set_cal_states(cls, values: dict[str, T]) -> dict[str, T]:
        id_value = values.get("cal_states")
        transition = values.get("transition")

        if id_value is None and transition is not None:
            values["cal_states"] = transition
        return values


# create additional options for spectroscopy
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

    use_cw: bool = Field(
        False, description="Perform a CW spectroscopy where no measure pulse is played."
    )
    spectroscopy_reset_delay: float = Field(
        1e-6, description="How long to wait after an acquisition in seconds."
    )
    acquisition_type: AcquisitionType = Field(
        AcquisitionType.SPECTROSCOPY,
        description="Acquisition type to use for the experiment.",
    )


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

    do_rotation: bool = Field(
        True,
        description="Whether to rotate the raw data based on calibration traces or "
        "principal component analysis.",
    )
    do_pca: bool = Field(
        False,
        description="Whether to perform principal component analysis on the raw data"
        " independent of whether there were calibration traces in the experiment.",
    )
    do_fitting: bool = Field(True, description="Whether to perform the fit.")
    fit_parameters_hints: dict | None = Field(
        None, description="Parameters hints accepted by lmfit."
    )
    save_figures: bool = Field(True, description="Whether to save the figures.")
    close_figures: bool = Field(True, description="Whether to close the figures.")


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

    do_fitting: bool = Field(True, description="Whether to perform the fit.")
    do_plotting: bool = Field(True, description="Whether to make plots.")
    do_raw_data_plotting: bool = Field(
        True, description="Whether to plot the raw data."
    )
    do_qubit_population_plotting: bool = Field(
        True, description="Whether to plot the qubit population."
    )


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

    do_analysis: bool = Field(True, description="Whether to run the analysis workflow.")
    update: bool = Field(
        False,
        description="Whether to update the setup based on the "
        "results from the analysis.",
    )


class QubitSpectroscopyExperimentOptions(BaseExperimentOptions):
    """Base options for the resonator spectroscopy experiment.

    Additional attributes:
        spectroscopy_reset_delay:
            How long to wait after an acquisition in seconds.
            Default: 1e-6.
    """

    spectroscopy_reset_delay: float = Field(
        1e-6, description="How long to wait after an acquisition in seconds."
    )


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

    do_fitting: bool = Field(True, description="Whether to perform the fit.")
    fit_parameters_hints: dict | None = Field(
        None, description="Parameters hints accepted by lmfit."
    )
    save_figures: bool = Field(True, description="Whether to save the figures.")
    close_figures: bool = Field(True, description="Whether to close the figures.")


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

    do_plotting: bool = Field(True, description="Whether to make plots.")
    do_raw_data_plotting: bool = Field(
        True, description="Whether to plot the raw data."
    )
    do_plotting_qubit_spectroscopy: bool = Field(
        True, description="Whether to plot the final qubit spectroscopy plot."
    )
