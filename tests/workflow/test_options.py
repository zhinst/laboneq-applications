from __future__ import annotations

from typing import Optional, TypeVar, Union

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
def r(a: int, options: A | B | None = None): ...
def neg_g(a: int, options: Union[A, B, None] = None): ...  # noqa: UP007
def neg_g2(a: int, options: Union[A, B] = None): ...  # noqa: UP007
def x(a: int, options: A): ...
def y(a: int, options: str): ...
def z(a: int, options: T | None = None): ...
def aa(a: int, options: AA | A = None): ...  # both AA and A are TaskBookOptions


class TestGetOptType:
    def test_get_valid_option(self):
        assert get_parameter_type(f) == A
        assert get_parameter_type(fbar) == A
        assert get_parameter_type(g) == A
        assert get_parameter_type(h) == A

    def test_get_invalid_option(self):
        assert get_parameter_type(noopt) is None
        assert get_parameter_type(notype) is None
        assert get_parameter_type(neg_g) is None
        assert get_parameter_type(neg_g2) is None
        assert get_parameter_type(r) is None
        assert get_parameter_type(x) is None
        assert get_parameter_type(y) is None
        assert get_parameter_type(z) is None
        assert get_parameter_type(aa) is None

    def test_failed(self):
        assert get_parameter_type(g) == A
