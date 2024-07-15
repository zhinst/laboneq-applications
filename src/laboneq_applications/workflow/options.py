"""Options for workflows."""

from __future__ import annotations

import sys
import typing
from typing import (
    Any,
    Callable,
    ForwardRef,
    Union,
    get_args,
    get_origin,
)

from laboneq_applications.core.options import BaseOptions, TuneupExperimentOptions


class TaskBookOptions(BaseOptions):
    """Base option for taskbook.

    Attributes:
        run_until:
            The task to run until.
            Default: `None`.
    """

    run_until: str | None = None


class TuneUpTaskBookOptions(TaskBookOptions):
    """Option class for tune-up taskbook.

    Attributes:
        create_experiment (TuneupExperimentOptions):
            The options for creating the experiment.
            Default: TuneupExperimentOptions().
    """

    create_experiment: TuneupExperimentOptions = TuneupExperimentOptions()


_PY_V39 = sys.version_info < (3, 10)

if not _PY_V39:
    from types import UnionType


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
            typing._eval_type(ForwardRef(arg), _globals, _locals) for arg in args
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
        opt_type = typing._eval_type(ForwardRef(type_hint), _globals, _locals)
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


def get_parameter_type(
    fn: Callable[[Any], Any],
    parameter: str = "options",
    type_check: type = TaskBookOptions,
) -> type[TaskBookOptions] | None:
    """Get the type of the parameter for a function-like object.

    The function-like object must have an `parameter` with a type hint,
    following the pattern `Union[Type, None]` or `Type | None` or `Optional[Type]`,
    where type must be of type `type_check`.

    Return:
        Type of the parameter if it exists and satisfies the above
        conditions, otherwise `None`.

    Raises:
        ValueError: When the type hint contains a subclass of `type_check`, but
        does not follow the pattern `Union[Type, None]` or `Type | None` or
        `Optional[Type]`.
    """
    expected_args_length = 2
    opt_type = _get_argument_types(fn, parameter)
    compatible_types = [
        t for t in opt_type if isinstance(t, type) and issubclass(t, type_check)
    ]

    if compatible_types:
        if len(opt_type) != expected_args_length or type(None) not in opt_type:
            raise ValueError(
                "It seems like you want to use the taskbook feature of automatically"
                "passing options to the tasks, but the type provided is wrong. "
                f"Please use either {compatible_types[0].__name__} | None, "
                f"Option[{compatible_types[0].__name__}] or "
                f"Union[{compatible_types[0].__name__},None]"
                "to enable this feature. Use any other type if you don't want to use"
                "this feature but still want pass options manually to the taskbook "
                "and its tasks.",
            )
        return compatible_types[0]
    return None
