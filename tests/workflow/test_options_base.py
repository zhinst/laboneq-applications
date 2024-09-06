from __future__ import annotations

import re
from typing import Literal

import pytest
from pydantic import Field, ValidationError

from laboneq_applications.workflow.options_base import BaseOptions


class OptionsUnderTest(BaseOptions):
    foo: int = 10
    bar: Literal["ge", "ef"] = "ge"
    fred: float = 1.0


class TestOptions:
    def test_create_options(self):
        opt = OptionsUnderTest()
        assert opt.foo == 10
        assert opt.bar == "ge"
        assert opt.fred == 1.0

    def test_not_accepting_extra_arguments(self):
        with pytest.raises(ValidationError):
            OptionsUnderTest(not_required=10)

    def test_invalid_input_raises_exception(self):
        with pytest.raises(ValidationError):
            OptionsUnderTest(count="could_not_parsed_to_int")

    def test_invalid_default(self):
        # default value is not valid
        class OptionsNotValid(BaseOptions):
            foo: int = "string"

        with pytest.raises(ValidationError):
            OptionsNotValid()

    def test_deserialization(self):
        input_options = {
            "foo": 10,
            "bar": "ge",
            "fred": 20,
        }
        opt = OptionsUnderTest(**input_options)
        serialized_dict = opt.to_dict()
        assert serialized_dict == {
            "foo": 10,
            "bar": "ge",
            "fred": 20,
        }

        # direct method of deserialization
        deserialized_options = opt.from_dict(serialized_dict)
        assert deserialized_options == opt

    def test_options_without_defaults(self):
        class OptionsWithoutDefaults(BaseOptions):
            foo: int
            bar: str = "ge"
            fred: float = 2.0

        with pytest.raises(ValidationError):
            OptionsWithoutDefaults(foo=1)

        class OptionsWithFieldDefaults(BaseOptions):
            foo: int = Field(default=10)

        opt = OptionsWithFieldDefaults()
        assert opt.foo == 10


def minify_string(s):
    return s.replace("\n", "").replace(" ", "").replace("│", "").replace("↳", "")


def strip_ansi_codes(s):
    """Remove all ANSI codes from the given string."""
    ansi_escape = re.compile(r"\x1b[^m]*m")
    return ansi_escape.sub("", s)


class TestOptionPrinting:
    class FullOption(BaseOptions):
        number: int = 1
        text: str = "This is a text"
        d: dict = Field(default_factory=dict)
        array: list = Field(default_factory=list)
        more: None | BaseOptions = None

    class SimpleOption(BaseOptions):
        a: int = 1

    def test_str(self):
        opt = self.FullOption(d={"a": 1}, array=[1, 2, 3])
        expected_str = (
            "FullOption(number=1,text='Thisisatext',d={'a':1},array=[1,2,3],more=None)"
        )
        assert expected_str == (strip_ansi_codes(minify_string(str(opt))))

        opt = self.FullOption(d={"a": 1}, array=[], more=self.SimpleOption())
        expected_str = (
            "FullOption(number=1,text='Thisisatext',d={'a':1},array=[],"
            "more=SimpleOption(a=1))"
        )

        assert expected_str == strip_ansi_codes(minify_string(str(opt)))

    def test_fmt(self):
        opt = self.FullOption(d={"a": 1}, array=[1, 2, 3])
        expected_repr = (
            "FullOption(number=1, text='This is a text', d={'a': 1}, array=[1, 2, 3], "
            "more=None)"
        )
        assert f"{opt}" == expected_repr
