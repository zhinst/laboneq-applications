from typing import Literal

import pytest
from laboneq.simple import AcquisitionType, AveragingMode, RepetitionMode
from pydantic import ValidationError

from laboneq_applications.core.options import (
    BaseExperimentOptions,
    BaseOptions,
    TuneupExperimentOptions,
)


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


class TestBaseExperimentOptions:
    def test_base_options(self):
        opt = BaseExperimentOptions()
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
        opt = BaseExperimentOptions(**input_options)
        assert opt.count == 10
        assert opt.acquisition_type == AcquisitionType.INTEGRATION
        assert opt.averaging_mode == AveragingMode.SEQUENTIAL
        assert opt.repetition_mode == RepetitionMode.FASTEST
        assert opt.repetition_time is None
        assert not opt.reset_oscillator_phase


class TestTuneupExperimentOptions:
    def test_create_options(self):
        # explicitly pass cal_states
        input_options = {
            "count": 2**12,
            "transition": "ge",
            "use_cal_traces": False,
            "cal_states": "gef",
        }
        opt = TuneupExperimentOptions(**input_options)
        assert opt.count == 2**12
        assert opt.transition == "ge"
        assert not opt.use_cal_traces
        assert opt.cal_states == "gef"

    def test_create_options_default_transition(self):
        # test cal_states different to default transition
        input_options = {
            "count": 2**12,
            "use_cal_traces": True,
            "cal_states": "ef",
        }
        opt = TuneupExperimentOptions(**input_options)
        assert opt.count == 2**12
        assert opt.transition == "ge"
        assert opt.use_cal_traces
        assert opt.cal_states == "ef"

    def test_create_options_default_cal_states(self):
        # test cal_states created from transition
        input_options = {
            "count": 2**12,
            "transition": "ef",
            "use_cal_traces": True,
        }
        opt = TuneupExperimentOptions(**input_options)
        assert opt.count == 2**12
        assert opt.transition == "ef"
        assert opt.use_cal_traces
        assert opt.cal_states == "ef"

    def test_create_options_default_transition_cal_states(self):
        # test default cal_states and transition
        input_options = {
            "count": 2**12,
            "use_cal_traces": True,
        }
        opt = TuneupExperimentOptions(**input_options)
        assert opt.count == 2**12
        assert opt.transition == "ge"
        assert opt.use_cal_traces
        assert opt.cal_states == "ge"
