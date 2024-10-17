"""Tests for laboneq_applications.qpu_types.tunable_transmon.qpu_types."""

import copy

import pytest

from laboneq_applications.qpu_types.tunable_transmon import (
    TunableTransmonQubit,
    TunableTransmonQubitParameters,
)

import tests.helpers.dsl as tsl


@pytest.fixture()
def q0(single_tunable_transmon_platform):
    return single_tunable_transmon_platform.qpu.qubits[0]


@pytest.fixture()
def multi_qubits(two_tunable_transmon_platform):
    return two_tunable_transmon_platform.qpu.qubits


class TestTunableTransmonQubit:
    def test_create(self):
        q = TunableTransmonQubit()
        assert isinstance(q.parameters, TunableTransmonQubitParameters)

    def test_transition_parameters_default(self, q0):
        drive_line, params = q0.transition_parameters()
        assert drive_line == "drive"
        assert params["amplitude_pi"] == 0.8

    def test_transition_parameters_ge(self, q0):
        drive_line, params = q0.transition_parameters("ge")
        assert drive_line == "drive"
        assert params["amplitude_pi"] == 0.8

    def test_transition_parameters_ef(self, q0):
        drive_line, params = q0.transition_parameters("ef")
        assert drive_line == "drive_ef"
        assert params["amplitude_pi"] == 0.7

    def test_transition_parameters_error(self, q0):
        with pytest.raises(ValueError) as err:
            q0.transition_parameters("gef")
        assert str(err.value) == "Transition 'gef' is not one of None, 'ge' or 'ef'."

    def test_default_integration_kernels(self, q0):
        assert q0.default_integration_kernels() == [
            tsl.pulse(function="const", amplitude=1, length=2e-6),
        ]

    def test_get_integration_kernels_default(self, q0):
        q0.parameters.readout_integration_kernels = "default"
        assert q0.get_integration_kernels() == [
            tsl.pulse(
                uid="__integration_kernel_q0_0",
                function="const",
                amplitude=1,
                length=2e-6,
            ),
        ]

    def test_get_integration_kernels_pulses_type_default(self, q0):
        q0.parameters.readout_integration_kernels = [
            {"function": "const", "amplitude": 2.0},
        ]
        q0.parameters.readout_integration_kernels_type = "default"
        assert q0.get_integration_kernels() == [
            tsl.pulse(
                uid="__integration_kernel_q0_0",
                function="const",
                amplitude=1,
                length=2e-6,
            ),
        ]

    def test_get_integration_kernels_pulses_type_optimal(self, q0):
        q0.parameters.readout_integration_kernels = [
            {"function": "const", "amplitude": 2.0},
        ]
        q0.parameters.readout_integration_kernels_type = "optimal"
        assert q0.get_integration_kernels() == [
            tsl.pulse(
                uid="__integration_kernel_q0_0",
                function="const",
                amplitude=2.0,
                length=1e-7,
            ),
        ]

    def test_get_integration_kernel_overrides(self, q0):
        assert q0.get_integration_kernels([{"function": "const"}]) == [
            tsl.pulse(
                uid="__integration_kernel_q0_0",
                function="const",
                amplitude=1,
                length=1e-7,
            ),
        ]

    def test_get_integration_kernels_invalid_overrides(self, q0):
        with pytest.raises(TypeError) as err:
            q0.get_integration_kernels({"function": "const"})

        assert str(err.value) == (
            "The readout integration kernels should be a list of pulse "
            "dictionaries or the values 'default' or 'optimal'. If no readout "
            "integration kernels have been specified, then the parameter "
            "TunableTransmonQubit.parameters.readout_integration_kernels_type'"
            " should be either 'default' or 'optimal'."
        )

    def test_get_integration_kernels_empty_list(self, q0):
        q0.parameters.readout_integration_kernels_type = "optimal"
        q0.parameters.readout_integration_kernels = []
        with pytest.raises(TypeError) as err:
            q0.get_integration_kernels()

        assert str(err.value) == (
            "TunableTransmonQubit.parameters.readout_integration_kernels' should be a "
            "list of pulse dictionaries."
        )

    def test_get_integration_kernels_invalid_kernel_pulses(self, q0):
        q0.parameters.readout_integration_kernels_type = "optimal"
        q0.parameters.readout_integration_kernels = "zoo"
        with pytest.raises(TypeError) as err:
            q0.get_integration_kernels()

        assert str(err.value) == (
            "TunableTransmonQubit.parameters.readout_integration_kernels' should be a "
            "list of pulse dictionaries."
        )

    def test_get_integration_kernels_invalid_kernels_type(self, q0):
        q0.parameters.readout_integration_kernels_type = "the_best"
        with pytest.raises(TypeError) as err:
            q0.get_integration_kernels()

        assert str(err.value) == (
            "The readout integration kernels should be a list of pulse "
            "dictionaries or the values 'default' or 'optimal'. If no readout "
            "integration kernels have been specified, then the parameter "
            "TunableTransmonQubit.parameters.readout_integration_kernels_type'"
            " should be either 'default' or 'optimal'."
        )

    def test_update(self, q0):
        q0.update({"readout_range_out": 10})
        assert q0.parameters.readout_range_out == 10

        q0.update({"readout_length": 10e-6})
        assert q0.parameters.readout_length == 10e-6

        # test update existing params but with None value
        q0.parameters.readout_pulse = None
        q0.update({"readout_pulse": {"function": "const"}})
        assert q0.parameters.readout_pulse == {"function": "const"}

        q0.update({"ge_drive_amplitude_pi": 0.1})
        assert q0.parameters.ge_drive_amplitude_pi == 0.1

        _original_ge_drive_pulse = copy.deepcopy(q0.parameters.ge_drive_pulse)
        q0.update({"ge_drive_pulse.beta": 0.5})
        assert q0.parameters.ge_drive_pulse["beta"] == 0.5
        assert (
            q0.parameters.ge_drive_pulse["function"]
            == _original_ge_drive_pulse["function"]
        )
        assert (
            q0.parameters.ge_drive_pulse["sigma"] == _original_ge_drive_pulse["sigma"]
        )

    def test_update_nonexisting_params(self, q0):
        # test that updating non-existing parameters raises an error

        original_params = copy.deepcopy(q0.parameters)
        with pytest.raises(ValueError) as err:
            q0.update(
                {
                    "readout_range_out": 10,
                    "non_existing_param": 10,
                    "readout_parameters.non_existing_param": 10,
                },
            )

        assert str(err.value) == (
            f"Cannot update {q0.uid}: Update parameters do not "
            f"match the qubit parameters: ['non_existing_param', "
            f"'readout_parameters.non_existing_param']."
        )
        # assert no parameters were updated
        assert q0.parameters == original_params

    def test_replace(self, q0):
        new_q0 = q0.replace(
            {
                "readout_range_out": 10,
                "readout_length": 10e-6,
                "ge_drive_amplitude_pi": 0.1,
            },
        )
        assert id(new_q0) != id(q0)
        assert new_q0.parameters.readout_range_out == 10
        assert new_q0.parameters.readout_length == 10e-6
        assert new_q0.parameters.ge_drive_amplitude_pi == 0.1

    def test_replace_wrong_params(self, q0):
        with pytest.raises(ValueError) as exc_info:
            _ = q0.replace(
                {
                    "wrong_param": 0,
                    "wrong_param_2": 1,
                },
            )
        assert str(exc_info.value) == f"Cannot update {q0.uid}"
        assert str(exc_info.value.__cause__) == (
            "Update parameters do not match the qubit "
            "parameters: ['wrong_param', 'wrong_param_2']"
        )

    def test_invalid_params_reported_correctly(self, q0):
        non_existing_params = [
            "non_existing_param",
            "readout_parameters.non_existing_param",
        ]
        with pytest.raises(ValueError) as err:
            q0.parameters._override(
                {
                    "readout_range_out": 10,
                    "non_existing_param": 10,
                    "readout_parameters.non_existing_param": 10,
                },
            )

        assert str(err.value) == (
            f"Update parameters do not match the qubit "
            f"parameters: {non_existing_params}"
        )

        # nested invalid parameters are reported correctly
        non_existing_params = [
            "drive_parameters_ge.non_existing.not_existing",
            "non_existing.not_existing",
        ]
        with pytest.raises(ValueError) as err:
            q0.parameters._override(
                {
                    "drive_parameters_ge.non_existing.not_existing": 10,
                    "non_existing.not_existing": 10,
                },
            )

        assert str(err.value) == (
            f"Update parameters do not match the qubit "
            f"parameters: {non_existing_params}"
        )

    @pytest.mark.parametrize("kernels_type", ["default", "optimal"])
    @pytest.mark.parametrize("thresholds", [None, [0, 0, 0]])
    def test_calibration(self, q0, thresholds, kernels_type):
        q0.parameters.readout_integration_discrimination_thresholds = thresholds
        q0.parameters.readout_integration_kernels_type = kernels_type
        qubit_calib = q0.calibration()
        acq_sig_calib = qubit_calib[q0.signals["acquire"]]
        assert acq_sig_calib.threshold == thresholds
        assert (
            acq_sig_calib.oscillator.frequency == 0
            if kernels_type == "optimal"
            else 100e6
        )


class TestTunableTransmonParameters:
    def test_create(self):
        p = TunableTransmonQubitParameters()

        assert p.readout_range_out == 5
        assert p.readout_range_in == 10
