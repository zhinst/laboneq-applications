from __future__ import annotations

import typing
from typing import Literal, Optional, TypeVar, Union

import pytest

from laboneq_applications.workflow.options import (
    WorkflowOptions,
)
from laboneq_applications.workflow.options_parser import (
    check_type,
    get_and_validate_param_type,
)


class A(WorkflowOptions): ...


class AA(WorkflowOptions): ...


class B: ...


T = TypeVar("T")


# allowed
def f(a: int, options: A | None = None): ...
def fbar(a: int, options: None | A = None): ...
def g(a: int, options: Union[A, None] = None): ...  # noqa: UP007
def h(a: int, options: Optional[A] = None): ...  # noqa: UP007


# not allowed
def noopt(a: int): ...
def notype(a: int, options): ...


def y(a: int, options: str): ...
def z(a: int, options: T | None = None): ...


# attempt to use workflow options in a wrong way
# when one of the types is WorkflowOptions, the option type should
# conform to A | None, Optional[A], or Union[A, None]
def nodefault(a: int, options: A | None): ...
def aa(a: int, options: AA | A = None): ...  # both AA and A are WorkflowOptions
def x(a: int, options: A): ...
def neg_g(a: int, options: Union[A, B, None] = None): ...  # noqa: UP007
def neg_g2(a: int, options: Union[A, B] = None): ...  # noqa: UP007
def r(a: int, options: A | B | None = None): ...


class TestGetOptType:
    def test_get_valid_option(self):
        assert get_and_validate_param_type(f, WorkflowOptions) == A
        assert get_and_validate_param_type(fbar, WorkflowOptions) == A
        assert get_and_validate_param_type(g, WorkflowOptions) == A
        assert get_and_validate_param_type(h, WorkflowOptions) == A

    def test_get_invalid_option(self):
        assert get_and_validate_param_type(noopt, WorkflowOptions) is None
        assert get_and_validate_param_type(notype, WorkflowOptions) is None
        assert get_and_validate_param_type(y, WorkflowOptions) is None
        assert get_and_validate_param_type(z, WorkflowOptions) is None

    def test_attempt_to_use_workflow_options_wrong(self):
        for fn in (nodefault, aa, x, neg_g, neg_g2, r):
            pytest.raises(TypeError, get_and_validate_param_type, fn)


class TestTypeValidator:
    def test_check_type(self):
        class A: ...

        class SubA(A): ...

        class B: ...

        def f(): ...

        # right type
        assert check_type(1, int, globals(), locals())
        assert check_type(A(), "A", globals(), locals())
        assert check_type(SubA(), "A", globals(), locals())
        assert check_type(1, "int | str", globals(), locals())
        assert check_type(1, "int | str | float", globals(), locals())
        assert check_type("1", "int | str", globals(), locals())
        assert check_type(1, Union[int, str], globals(), locals())
        assert check_type(1, object, globals(), locals())
        assert check_type(1, typing.Any, globals(), locals())

        assert check_type([1, 2], list, globals(), locals())
        assert check_type({1, 2}, set, globals(), locals())
        assert check_type((1, 2), tuple, globals, locals())

        assert check_type({"a": 1}, dict, globals(), locals())
        assert check_type(None, "None", globals(), locals())
        assert check_type(None, Optional[int], globals(), locals())
        assert check_type(1, Optional[int], globals(), locals())

        assert check_type("ge", Literal["ge", "be"], globals(), locals())

        assert check_type(f, "Callable", globals(), locals())

        # invalid type
        assert not check_type(1, str, globals(), locals())
        assert not check_type(1.0, int, globals(), locals())
        assert not check_type(A(), "int | str", globals(), locals())
        assert not check_type(A(), int, globals(), locals())
        assert not check_type([1, 2, 3], set, globals(), locals())
        assert not check_type([1, 2, 3, 4], dict[int], globals(), locals())
        assert not check_type(B(), "A", globals(), locals())
