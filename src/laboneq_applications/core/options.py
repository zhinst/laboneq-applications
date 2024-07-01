"""Options for the experiment tasks."""

from __future__ import annotations

from typing import Annotated, Literal, TypeVar

from laboneq.simple import AcquisitionType, AveragingMode, RepetitionMode
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

NonNegativeInt = Annotated[int, Field(ge=0)]
T = TypeVar("T")


class BaseOptions(BaseModel):
    """Base class for all Option classes."""

    model_config = ConfigDict(
        validate_assignment=True,
        validate_default=True,
        extra="forbid",
    )

    def to_dict(self) -> dict:
        """Generate a dictionary representation of the options."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> BaseOptions:
        """Create an instance of the class from a dictionary."""
        return cls(**data)

    def __eq__(self, other: BaseOptions) -> bool:
        """Check if two options are equal."""
        if not isinstance(other, BaseOptions):
            return False
        return self.to_dict() == other.to_dict()


class BaseExperimentOptions(BaseOptions):
    """Base options for the experiment.

    Attributes:
        count:
            The number of repetitions.
            Default: A common choice in practice, 4096.
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

    count: NonNegativeInt = 4096
    acquisition_type: str | AcquisitionType = AcquisitionType.INTEGRATION
    averaging_mode: str | AveragingMode = AveragingMode.CYCLIC
    repetition_mode: str | RepetitionMode = RepetitionMode.FASTEST
    repetition_time: float | None = None
    reset_oscillator_phase: bool = False

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

    transition: Literal["ge", "ef"] = "ge"
    use_cal_traces: bool = True
    cal_states: str | tuple = "ge"

    @model_validator(mode="before")
    @classmethod
    def _set_cal_states(cls, values: dict[str, T]) -> dict[str, T]:
        id_value = values.get("cal_states")
        transition = values.get("transition")

        if id_value is None and transition is not None:
            values["cal_states"] = transition
        return values


class TaskBookOptions(BaseOptions):
    """Base option for taskbook.

    Attributes:
        run_until:
            The task to run until.
            Default: `None`.
    """

    run_until: str | None = None
