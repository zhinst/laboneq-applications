from __future__ import annotations

from typing import Optional, TypeVar, Union

import pytest

from laboneq_applications.workflow.options import (
    TaskBookOptions,
    get_parameter_type,
)


class A(TaskBookOptions): ...


class AA(TaskBookOptions): ...


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


# attempt to use taskbook options in a wrong way
# when one of the types is TaskBookOptions, the option type should
# conform to A | None, Optional[A], or Union[A, None]
def aa(a: int, options: AA | A = None): ...  # both AA and A are TaskBookOptions
def x(a: int, options: A): ...
def neg_g(a: int, options: Union[A, B, None] = None): ...  # noqa: UP007
def neg_g2(a: int, options: Union[A, B] = None): ...  # noqa: UP007
def r(a: int, options: A | B | None = None): ...


class TestGetOptType:
    def test_get_valid_option(self):
        assert get_parameter_type(f) == A
        assert get_parameter_type(fbar) == A
        assert get_parameter_type(g) == A
        assert get_parameter_type(h) == A

    def test_get_invalid_option(self):
        assert get_parameter_type(noopt) is None
        assert get_parameter_type(notype) is None
        assert get_parameter_type(y) is None
        assert get_parameter_type(z) is None

    def test_attempt_to_use_taskbook_options_wrong(self):
        for fn in (aa, x, neg_g, neg_g2, r):
            pytest.raises(ValueError, get_parameter_type, fn)
