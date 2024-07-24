"""Options for the experiment tasks."""

from __future__ import annotations

from io import StringIO
from typing import Annotated, Literal, TypeVar, final

from laboneq.simple import AcquisitionType, AveragingMode, RepetitionMode
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticUndefinedType
from rich.console import Console
from rich.pretty import pprint

NonNegativeInt = Annotated[int, Field(ge=0)]
T = TypeVar("T")


class BaseOptions(BaseModel):
    """Base class for all Option classes."""

    model_config = ConfigDict(
        validate_assignment=True,
        validate_default=True,
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def _check_defaults(cls, data):  # noqa: ANN206, ANN001
        for name, field in cls.model_fields.items():
            if (
                isinstance(field.default, PydanticUndefinedType)
                and field.default_factory is None
            ):
                raise ValueError(f"Field {name!r} does not have a default value")
        return data

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

    @final
    def __str__(self):
        with StringIO() as buffer:
            console = Console(file=buffer)
            pprint(self, console=console, expand_all=True, indent_guides=True)
            return buffer.getvalue()

    @final
    def __format__(self, _):  # noqa: ANN001
        return self.__repr__()

    @final
    def _repr_pretty_(self, p, _cycle):  # noqa: ANN001, ANN202
        # For Notebooks
        p.text(str(self))


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
