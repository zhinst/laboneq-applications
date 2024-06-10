"""Options for the experiment tasks."""

from __future__ import annotations

from typing import Annotated

from laboneq.simple import AcquisitionType, AveragingMode, RepetitionMode
from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator

NonNegativeInt = Annotated[int, Field(ge=0)]


class BaseOptions(BaseModel):
    """Base class for all Option classes."""

    model_config = ConfigDict(validate_assignment=True, validate_default=True)

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
        count (NonNegativeInt):
            The number of repetitions.
            Default: A common choice in practice, 4096.
        averaging_mode (AveragingMode):
            Averaging mode to use for the experiment.
            Default: `AveragingMode.CYCLIC`.
        acquisition_type (AcquisitionType):
            Acquisition type to use for the experiment.
            Default: `AcquisitionType.INTEGRATION`.
        repetition_mode (str | RepetitionMode):
            The repetition mode to use for the experiment.
            Default: `RepetitionMode.FASTEST`.
        repetition_time (float | None):
            The repetition time.
            Default: None.
        reset_oscillator_phase (bool):
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


def create_validate_opts(
    input_options: dict | None,
    custom_options: dict | None = None,
    base: BaseModel = BaseOptions,
) -> BaseModel:
    """Create an options template (model) and validate the input options against it.

    The template contains options from the base model. Custom options, if provided, are
    added and will override the base options if the field names are the same.

    Arguments:
        input_options:
            The input options.
        custom_options:
            Options to define the model, in addition to the base options.
            The keys are the field names and the values are tuples of the form
            (type, default_value) or (type, ...), where ... indicates that the
            field is required.
            Override the base options if the field names are the same.
            If None, no additional options are added to the base option.
        base:
            The base option model to built upon. Default: BaseOptions.

    Returns:
        BaseModel: The validated options.

    Example:
        ```python
        class ExampleOption(BaseOptions):
            foo: int
            bar: str
        custom_options = {
            "fred": (int, ...),
            "default_fed": (str, "fed"),
        }
        options = {
            "foo": 10,
            "bar": "ge",
            "fred": 20,
        }
        opt = create_validate_opts(options, custom_options, base=BaseOptions)
        ```
    """
    if input_options is None:
        input_options = {}
    if custom_options is None:
        custom_options = {}
    option_model = create_model(
        "option_model",
        **custom_options,
        __base__=base,
        __module__="laboneq_applications.core.options",
    )
    return option_model(**input_options)
