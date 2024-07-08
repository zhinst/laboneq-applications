"""Options for the experiment tasks."""

from __future__ import annotations

import sys
import typing
from io import StringIO
from typing import (
    Annotated,
    Any,
    Callable,
    Literal,
    TypeVar,
    Union,
    final,
    get_args,
    get_origin,
)

from laboneq.simple import AcquisitionType, AveragingMode, RepetitionMode
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from rich.console import Console
from rich.pretty import pprint

_PY_V39 = sys.version_info < (3, 10)

if not _PY_V39:
    from types import UnionType

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


class TaskBookOptions(BaseOptions):
    """Base option for taskbook.

    Attributes:
        run_until:
            The task to run until.
            Default: `None`.
    """

    run_until: str | None = None


def _get_argument_types(
    fn: Callable,
    arg_name: str,
) -> set[type]:
    """Get the type of the parameter for a function-like object.

    Return:
        Set of the type of the parameter. Empty set if the parameter
        does not exist or does not have a type hint.
    """
    _globals = getattr(fn, "__globals__", {})
    _locals = _globals

    # typing.get_type_hints does not work on 3.9 with A | None = None.
    # It is also overkill for retrieving type of a single parameter of
    # only function-like objects, and will cause problems
    # when other parameters have no type hint or type hint imported
    # conditioned on type_checking.
    hint = getattr(fn, "__annotations__", {}).get(arg_name, None)
    if hint is None:
        return set()

    if _PY_V39:
        return _get_argument_types_v39(hint, _globals, _locals)

    return _parse_types(hint, _globals, _locals, _PY_V39)


def _get_argument_types_v39(hint: str | type, _globals, _locals) -> set[type]:  # noqa: ANN001
    return_types: set[type]
    args = hint.split("|")
    if len(args) > 1:
        args = [arg.strip() for arg in args]
        return_types = {
            typing._eval_type(typing.ForwardRef(arg), _globals, _locals) for arg in args
        }
        return return_types

    return _parse_types(hint, _globals, _locals, is_py_39=True)


def _parse_types(
    type_hint: str | type,
    _globals,  # noqa: ANN001
    _locals,  # noqa: ANN001
    is_py_39: bool,  # noqa: FBT001
) -> set[type]:
    if isinstance(type_hint, str):
        opt_type = typing._eval_type(typing.ForwardRef(type_hint), _globals, _locals)
    else:
        opt_type = type_hint
    if _is_union_type(opt_type, is_py_39):
        return set(get_args(opt_type))
    return {opt_type}


def _is_union_type(opt_type: type, is_py_39: bool) -> bool:  # noqa: FBT001
    if (
        is_py_39
        and get_origin(opt_type) == Union
        or (not is_py_39 and get_origin(opt_type) in (UnionType, Union))
    ):
        return True
    return False


def get_option_type(
    fn: Callable[[Any], Any],
    type_check: type = TaskBookOptions,
) -> type[TaskBookOptions] | None:
    """Get the type of the options parameter for a function-like object.

    The function-like object must have an options parameter with a type hint,
    following the pattern `Union[Type, None]` or `Type | None` or `Optional[Type]`,
    where `Type` must be a subclass of `TaskBookOptions`.

    Return:
        Type of the options parameter if it exists and satisfies the above
        conditions, otherwise `None`.
    """
    expected_args_length = 2
    opt_type = _get_argument_types(fn, "options")
    if len(opt_type) != expected_args_length or type(None) not in opt_type:
        return None

    for t in opt_type:
        if isinstance(t, type) and issubclass(t, type_check):  # ignore typevars
            return t
    return None
