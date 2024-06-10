from typing import Literal

import pytest
from laboneq.simple import AcquisitionType, AveragingMode, RepetitionMode
from pydantic import ValidationError

from laboneq_applications.core.options import (
    BaseExperimentOptions,
    BaseOptions,
    create_validate_opts,
)


class OptionsUnderTest(BaseOptions):
    foo: int
    bar: Literal["ge", "ef"]
    optional_field: float = 1.0


class TestOptions:
    def test_create_options(self):
        custom_fields = {"more_foo": (int, ...), "optional_field2": (float, 2.0)}
        input_options = {
            "foo": 10,
            "bar": "ge",
            "more_foo": 20,
        }
        opt = create_validate_opts(input_options, custom_fields, base=OptionsUnderTest)
        assert opt.foo == 10
        assert opt.bar == "ge"
        assert opt.optional_field == 1.0
        assert opt.more_foo == 20
        assert opt.optional_field2 == 2.0

    def test_without_custom_fields(self):
        input_options = {
            "foo": 10,
            "bar": "ge",
            "optional_field": 1.0,
        }
        opt = create_validate_opts(input_options, {}, base=OptionsUnderTest)
        assert opt.foo == 10
        assert opt.bar == "ge"
        assert opt.optional_field == 1.0

    def test_tolerate_extra_arguments(self):
        input_options = {
            "foo": 10,
            "bar": "ge",
            "optional_field": 1.0,
            "not_required": "not_required",
        }
        opt = create_validate_opts(input_options, {}, base=OptionsUnderTest)
        assert opt.foo == 10
        assert opt.bar == "ge"
        assert opt.optional_field == 1.0
        assert not hasattr(opt, "not_required")

    def test_missing_required(self):
        custom_fields = {"more_foo": (int, ...), "optional_field2": (float, 2.0)}
        input_options = {
            "bar": "ge",
            "optional_field": 1.0,
            "more_foo": 20,
        }
        with pytest.raises(ValidationError):
            create_validate_opts(input_options, custom_fields, base=OptionsUnderTest)

        input_options = {
            "foo": 10,
            "bar": "ge",
            "optional_field": 1.234,
        }
        with pytest.raises(ValidationError):
            create_validate_opts(input_options, custom_fields, base=OptionsUnderTest)

    def test_invalid_input_raises_exception(self):
        # wrong type for field1
        custom_fields = {"more_foo": (int, ...), "optional_field2": (float, 2.0)}
        input_options = {
            "foo": "cannot_parse_to_int",
            "bar": "ge",
            "more_foo": 20,
        }

        with pytest.raises(ValidationError):
            create_validate_opts(input_options, custom_fields, base=OptionsUnderTest)

        # wrong values for field with Literal type
        input_options = {
            "foo": 10,
            "bar": "abc",
            "more_foo": 20,
        }
        with pytest.raises(ValidationError):
            create_validate_opts(input_options, custom_fields, base=OptionsUnderTest)

    def test_override(self):
        # override the base field optional_field
        custom_fields = {"more_foo": (int, ...), "optional_field": (str, "override")}
        input_options = {
            "foo": 10,
            "bar": "ge",
            "more_foo": 20,
        }
        opt = create_validate_opts(input_options, custom_fields, base=OptionsUnderTest)
        assert opt.foo == 10
        assert opt.bar == "ge"
        assert opt.optional_field == "override"
        assert opt.more_foo == 20

    def test_validate_default(self):
        # default value is not valid
        custom_fields = {
            "more_foo": (int, ...),
            "optional_field2": (float, "not_a_float"),
        }
        input_options = {
            "foo": 10,
            "bar": "ge",
            "more_foo": 20,
        }
        with pytest.raises(ValidationError):
            create_validate_opts(input_options, custom_fields, base=OptionsUnderTest)

    def test_deserialization(self):
        custom_fields = {"more_foo": (int, ...), "optional_field2": (float, 2.0)}
        input_options = {
            "foo": 10,
            "bar": "ge",
            "more_foo": 20,
        }
        opt = create_validate_opts(input_options, custom_fields, base=OptionsUnderTest)
        serialized_dict = opt.to_dict()
        assert serialized_dict == {
            "foo": 10,
            "bar": "ge",
            "optional_field": 1.0,
            "more_foo": 20,
            "optional_field2": 2.0,
        }
        deserialized_options = create_validate_opts(
            serialized_dict,
            custom_fields,
            base=OptionsUnderTest,
        )
        assert deserialized_options == opt

        # direct method of deserialization
        deserialized_options = opt.from_dict(serialized_dict)
        assert deserialized_options == opt


@pytest.mark.parametrize("empty_options", [{}, None])
class TestBaseExperimentOptions:
    @pytest.fixture(autouse=True)
    def _test_base_options(self, empty_options):
        opt = create_validate_opts(empty_options, base=BaseExperimentOptions)
        assert opt.count == 4096
        assert opt.acquisition_type == AcquisitionType.INTEGRATION
        assert opt.averaging_mode == AveragingMode.CYCLIC
        assert opt.repetition_mode == RepetitionMode.FASTEST
        assert opt.repetition_time is None
        assert not opt.reset_oscillator_phase

    def test_create_options(self):
        input_options = {
            "count": 10,
            "acquisition_type": AcquisitionType.INTEGRATION,
            "averaging_mode": "sequential",
        }
        opt = create_validate_opts(input_options, base=BaseExperimentOptions)
        assert opt.count == 10
        assert opt.acquisition_type == AcquisitionType.INTEGRATION
        assert opt.averaging_mode == AveragingMode.SEQUENTIAL
        assert opt.repetition_mode == RepetitionMode.FASTEST
        assert opt.repetition_time is None
        assert not opt.reset_oscillator_phase

    def test_create_options_with_extra_fields(self):
        custom_fields = {
            "extra_field": (int, ...),
            "active_reset": (bool, True),
        }
        input_options = {
            "count": 10,
            "acquisition_type": "integration_trigger",
            "averaging_mode": "cyclic",
            "extra_field": 20,
            "extra_optional_field": "extra",
        }
        opt = create_validate_opts(
            input_options,
            custom_fields,
            base=BaseExperimentOptions,
        )
        assert opt.count == 10
        assert opt.acquisition_type == AcquisitionType.INTEGRATION
        assert opt.averaging_mode == AveragingMode.CYCLIC
        assert opt.repetition_mode == RepetitionMode.FASTEST
        assert opt.repetition_time is None
        assert not opt.reset_oscillator_phase
        assert opt.extra_field == 20
        assert opt.active_reset
