"""Options parser for workflows."""

from __future__ import annotations

import inspect
import sys
import typing
from typing import Any, Callable, ForwardRef, Union, get_args, get_origin

from laboneq_applications.workflow.options_base import BaseOptions

_PY_V39 = sys.version_info < (3, 10)

if not _PY_V39:
    from types import UnionType


def _get_argument_types(
    fn: Callable[..., object],
    arg_name: str,
) -> set[type]:
    """Get the type of the parameter for a function-like object.

    Return:
        A set containing the parameter's types. Empty set if the parameter
        does not exist or does not have a type hint.
    """
    _globals = getattr(fn, "__globals__", {})
    _locals = _globals

    # typing.get_type_hints does not work on 3.9 with A | None = None.
    # It is also overkill for retrieving type of a single parameter of
    # only function-like objects, and will cause problems
    # when other parameters have no type hint or type hint imported
    # conditioned on type_checking.

    param = inspect.signature(fn).parameters.get(arg_name, None)
    if param is None or param.annotation is inspect.Parameter.empty:
        return set()

    if _PY_V39:
        return _get_argument_types_v39(param.annotation, _globals, _locals)

    return _parse_types(param.annotation, _globals, _locals, is_py_39=False)


def _get_default_argument(
    fn: Callable,
    arg_name: str,
) -> Any:  # noqa: ANN401
    """Get the default value of the parameter for a function-like object."""
    param = inspect.signature(fn).parameters.get(arg_name, None)
    if param is None:
        return inspect.Parameter.empty

    return param.default


def _get_argument_types_v39(
    hint: str | type,
    _globals: dict,
    _locals: dict,
) -> set[type]:
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
    *,
    is_py_39: bool,
) -> set[type]:
    if isinstance(type_hint, str):
        opt_type = typing._eval_type(ForwardRef(type_hint), _globals, _locals)
    else:
        opt_type = type_hint
    if _is_union_type(opt_type, is_py_39):
        return set(get_args(opt_type))
    return {opt_type}


def _is_union_type(opt_type: type, is_py_39: bool) -> bool:  # noqa: FBT001
    return (
        is_py_39
        and get_origin(opt_type) == Union
        or (not is_py_39 and get_origin(opt_type) in (UnionType, Union))
    )


T = typing.TypeVar("T")


def _unwrap_wrapped_func(func: Callable) -> Callable:
    """Unwrap wrappers from a function."""
    while hasattr(func, "__wrapped__"):
        func = func.__wrapped__
    return func


def get_and_validate_param_type(
    fn: Callable,
    parameter: str = "options",
    type_check: type[T] = BaseOptions,
) -> type[T] | None:
    """Get the type of the parameter for a function-like object.

    The function-like object must have an `parameter` with a type hint,
    following any of the following patterns:

        * `Union[type, None]`
        * `type | None`
        * `Optional[type]`.

    Returns:
        Type of the parameter if it exists and satisfies the above
        conditions, otherwise `None`.

    Raises:
        ValueError: When the type hint contains a subclass of `type_check`, but
            does not follow any of the specific patterns.
    """
    expected_args_length = 2
    opt_type = _get_argument_types(_unwrap_wrapped_func(fn), parameter)
    opt_default = _get_default_argument(_unwrap_wrapped_func(fn), parameter)
    compatible_types = [
        t for t in opt_type if isinstance(t, type) and issubclass(t, type_check)
    ]

    if compatible_types:
        if (
            len(opt_type) != expected_args_length
            or type(None) not in opt_type
            or opt_default is not None
        ):
            raise TypeError(
                "It seems like you want to use the workflow feature of automatically "
                "passing options to the tasks, but the type provided is wrong. "
                f"Please use either {compatible_types[0].__name__} | None = None, "
                f"Optional[{compatible_types[0].__name__}] = None or "
                f"Union[{compatible_types[0].__name__},None] = None "
                "to enable this feature. Use any other type if you don't want to use "
                "this feature but still want pass options manually to the workflow "
                "and its tasks.",
            )
        return compatible_types[0]
    return None
