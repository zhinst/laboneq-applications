from __future__ import annotations

import pytest

from laboneq_applications.workflow.options_base import (
    BaseOptions,
    option_field,
    options,
)

from tests.helpers.format import minify_string, strip_ansi_codes


@options
class OptionsUnderTest(BaseOptions):
    foo: int = 10
    bar: int | str = option_field("ge")
    fred: float = option_field(1.0)


class TestOptions:
    def test_create_options(self):
        opt = OptionsUnderTest()
        assert opt.foo == 10
        assert opt.bar == "ge"
        assert opt.fred == 1.0

    def test_not_accepting_extra_arguments(self):
        with pytest.raises(TypeError):
            OptionsUnderTest(not_required=10)

    def test_invalid_input_raises_exception(self):
        with pytest.raises(TypeError):
            OptionsUnderTest(count="could_not_parsed_to_int")

    @pytest.mark.xfail(
        reason="Remove when type checking is enabled for the new options"
    )
    def test_invalid_default(self):
        # default value is not valid
        @options
        class OptionsNotValid(BaseOptions):
            foo: int = option_field("string")

        with pytest.raises(TypeError):
            OptionsNotValid()

    def test_serialization(self):
        opt = OptionsUnderTest()
        d = opt.to_dict()
        assert d == {
            "foo": 10,
            "bar": "ge",
            "fred": 1.0,
        }

        @options
        class OptionsWithIgnoreField(BaseOptions):
            foo: int = option_field(10)
            bar: int | str = option_field("ge", exclude=True)
            fred: float = option_field(1.0)

        opt = OptionsWithIgnoreField()
        d = opt.to_dict()
        assert d == {
            "foo": 10,
            "fred": 1.0,
        }

    def test_deserialization(self):
        input_options = {
            "foo": 10,
            "bar": "ge",
            "fred": 20.0,
        }
        loaded_opt = OptionsUnderTest.from_dict(input_options)
        assert loaded_opt.foo == 10
        assert loaded_opt.bar == "ge"
        assert loaded_opt.fred == 20.0

        # direct method of deserialization
        loaded_opt = OptionsUnderTest(**input_options)
        assert loaded_opt.foo == 10
        assert loaded_opt.bar == "ge"
        assert loaded_opt.fred == 20.0

    def test_eq(self):
        opt1 = OptionsUnderTest()
        opt2 = OptionsUnderTest(foo=10, bar="ge", fred=1.0)
        assert opt1 == opt2

        opt2.foo = 9
        assert opt1 != opt2

        @options
        class AnotherOption(BaseOptions): ...

        assert opt1 != AnotherOption()

    def test_options_without_defaults(self):
        class OptionsWithoutDefaults(BaseOptions):
            foo: int
            bar: str = "ge"
            fred: float = 2.0

        with pytest.raises(TypeError):
            OptionsWithoutDefaults(foo=1)

        @options
        class OptionsWithFieldDefaults(BaseOptions):
            foo: int = option_field(default=10)

        opt = OptionsWithFieldDefaults()
        assert opt.foo == 10


@options
class FullOption(BaseOptions):
    number: int = option_field(1)
    text: str = option_field("This is a text")
    d: dict = option_field(factory=dict)
    array: list = option_field(factory=list)
    more: None | BaseOptions = option_field(None)


@options
class SimpleOption(BaseOptions):
    a: int = option_field(1)


class TestOptionPrinting:
    def test_str(self):
        opt = FullOption(d={"a": 1}, array=[1, 2, 3])
        expected_str = (
            "FullOption(number=1,text='Thisisatext',d={'a':1},array=[1,2,3],more=None)"
        )
        assert expected_str == (strip_ansi_codes(minify_string(str(opt))))
        opt = FullOption(d={"a": 1}, array=[], more=SimpleOption())
        expected_str = (
            "FullOption(number=1,text='Thisisatext',d={'a':1},array=[],"
            "more=SimpleOption(a=1))"
        )
        assert expected_str == strip_ansi_codes(minify_string(str(opt)))

    def test_fmt(self):
        opt = FullOption(d={"a": 1}, array=[1, 2, 3])
        expected_str = (
            "FullOption(number=1,text='Thisisatext',d={'a':1},array=[1,2,3],"
            "more=None)"
        )
        assert expected_str == strip_ansi_codes(minify_string(str(f"{opt}")))
