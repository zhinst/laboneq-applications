from __future__ import annotations

from typing import Optional, TypeVar, Union

import pytest

from laboneq_applications.workflow.options import (
    WorkflowOptions,
)
from laboneq_applications.workflow.options_parser import get_and_validate_param_type


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
        assert get_and_validate_param_type(f) == A
        assert get_and_validate_param_type(fbar) == A
        assert get_and_validate_param_type(g) == A
        assert get_and_validate_param_type(h) == A

    def test_get_invalid_option(self):
        assert get_and_validate_param_type(noopt) is None
        assert get_and_validate_param_type(notype) is None
        assert get_and_validate_param_type(y) is None
        assert get_and_validate_param_type(z) is None

    def test_attempt_to_use_workflow_options_wrong(self):
        for fn in (nodefault, aa, x, neg_g, neg_g2, r):
            pytest.raises(TypeError, get_and_validate_param_type, fn)
