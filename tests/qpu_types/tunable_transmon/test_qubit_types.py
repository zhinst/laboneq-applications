"""Tests for laboneq_applications.qpu_types.tunable_transmon.qpu_types."""

import copy

import pytest

import tests.helpers.dsl as tsl
from laboneq_applications.qpu_types.tunable_transmon import (
    TunableTransmonQubit,
    TunableTransmonQubitParameters,
)


@pytest.fixture()
def q0(single_tunable_transmon):
    return single_tunable_transmon.qubits[0]


@pytest.fixture()
def multi_qubits(two_tunable_transmon):
    return two_tunable_transmon.qubits


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
        q0.parameters.readout_integration_parameters["kernels"] = "default"
        assert q0.get_integration_kernels() == [
            tsl.pulse(
                uid="__integration_kernel_q0_0",
                function="const",
                amplitude=1,
                length=2e-6,
            ),
        ]

    def test_get_integration_kernels_pulses(self, q0):
        q0.parameters.readout_integration_parameters["kernels"] = [
            {"function": "const", "amplitude": 2.0},
        ]
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

    def test_get_integration_kernels_empty_list(self, q0):
        q0.parameters.readout_integration_parameters["kernels"] = []
        with pytest.raises(TypeError) as err:
            q0.get_integration_kernels()

        assert str(err.value) == (
            "TunableTransmonQubit readout integration kernels"
            " should be either 'default' or a list of pulse dictionaries."
        )

    def test_get_integration_kernels_invalid_type(self, q0):
        q0.parameters.readout_integration_parameters["kernels"] = "zoo"
        with pytest.raises(TypeError) as err:
            q0.get_integration_kernels()

        assert str(err.value) == (
            "TunableTransmonQubit readout integration kernels"
            " should be either 'default' or a list of pulse dictionaries."
        )

    def test_update(self, q0):
        q0.update({"readout_range_out": 10})
        assert q0.parameters.readout_range_out == 10

        q0.update({"readout_parameters": {"length": 10e-6}})
        assert q0.parameters.readout_parameters["length"] == 10e-6

        # test update existing params but with None value
        q0.parameters.readout_parameters["pulse"] = None
        q0.update({"readout_parameters.pulse": {"function": "const"}})
        assert q0.parameters.readout_parameters["pulse"] == {"function": "const"}

        _original_drive_parameters_ge = copy.deepcopy(q0.parameters.drive_parameters_ge)
        q0.update({"drive_parameters_ge.amplitude_pi": 0.1})
        assert q0.parameters.drive_parameters_ge["amplitude_pi"] == 0.1
        assert (
            q0.parameters.drive_parameters_ge["amplitude_pi2"]
            == _original_drive_parameters_ge["amplitude_pi2"]
        )
        assert (
            q0.parameters.drive_parameters_ge["length"]
            == _original_drive_parameters_ge["length"]
        )
        assert (
            q0.parameters.drive_parameters_ge["pulse"]
            == _original_drive_parameters_ge["pulse"]
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

        assert str(err.value) == f"Cannot update {q0.uid}"
        # assert no parameters were updated
        assert q0.parameters == original_params

    def test_replace(self, q0):
        new_q0 = q0.replace(
            {
                "readout_range_out": 10,
                "readout_parameters": {"length": 10e-6},
                "drive_parameters_ge.amplitude_pi": 0.1,
            },
        )
        assert id(new_q0) != id(q0)
        assert new_q0.parameters.readout_range_out == 10
        assert new_q0.parameters.readout_parameters["length"] == 10e-6
        assert new_q0.parameters.drive_parameters_ge["amplitude_pi"] == 0.1

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


class TestTunableTransmonParameters:
    def test_create(self):
        p = TunableTransmonQubitParameters()

        assert p.readout_range_out == 5
        assert p.readout_range_in == 10
