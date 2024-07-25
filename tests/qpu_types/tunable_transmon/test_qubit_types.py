"""Tests for laboneq_applications.qpu_types.tunable_transmon.qpu_types."""

import copy

import pytest

import tests.helpers.dsl as tsl
from laboneq_applications.qpu_types.tunable_transmon import (
    TunableTransmonQubit,
    TunableTransmonQubitParameters,
    modify_qubits,
    modify_qubits_context,
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


class TestOverrideParameters:
    def _equal_except(self, q0, q0_temp, key):
        for k, v in q0.__dict__.items():
            if k not in ("parameters"):
                assert getattr(q0_temp, k) == v
        for k, v in q0.parameters.__dict__.items():
            if k == key:
                continue
            assert getattr(q0_temp.parameters, k) == v

    def test_create_temp_single_qubit(self, q0):
        original_q0 = copy.deepcopy(q0)
        [q0_temp] = modify_qubits([(q0, {"readout_range_out": 10})])
        assert q0_temp.parameters.readout_range_out == 10
        assert original_q0 == q0
        self._equal_except(q0, q0_temp, "readout_range_out")

        q0.parameters.nested_params = {"a": {"a": 2}, "b": 1}
        # replacing the nested dictionary
        [q0_temp] = modify_qubits([(q0, {"nested_params": {"a": 2}})])
        assert q0_temp.parameters.nested_params == {"a": 2}
        assert original_q0 == q0
        self._equal_except(q0, q0_temp, "nested_params")

        # replacing partially the nested dictionary
        [q0_temp] = modify_qubits([(q0, {"nested_params.a.a": 3})])
        assert q0_temp.parameters.nested_params == {"a": {"a": 3}, "b": 1}
        assert original_q0 == q0
        self._equal_except(q0, q0_temp, "nested_params")

        # replacing a real nested parameter of qubits
        [q0_temp] = modify_qubits(
            [(q0, {"drive_parameters_ge.pulse.beta": 0.1})],
        )
        assert q0_temp.parameters.drive_parameters_ge["pulse"]["beta"] == 0.1
        assert original_q0 == q0
        self._equal_except(q0, q0_temp, "drive_parameters_ge")

    def test_nonexisting_params(self, q0):
        # test that updating non-existing parameters raises an error
        original_params = copy.deepcopy(q0)
        with pytest.raises(ValueError) as err:
            modify_qubits(
                [
                    (
                        q0,
                        {
                            "non_existing_param": 10,
                            "readout_parameters.non_existing_param": 10,
                            "readout_range_out": 10,
                        },
                    ),
                ],
            )

        assert str(err.value) == f"Cannot update {q0.uid}"
        # assert no parameters were updated
        assert q0 == original_params

    def test_return_same_qubits(self, multi_qubits):
        q0, q1 = multi_qubits
        [q0_temp] = modify_qubits([(q0, {})])
        assert q0_temp == q0

        [q0_temp, q1_temp] = modify_qubits([(q0, {}), (q1, {})])
        assert q0_temp == q0
        assert q1_temp == q1

    def test_override_multiple_qubits(self, multi_qubits):
        q0, q1 = multi_qubits
        original_q0 = copy.deepcopy(q0)
        original_q1 = copy.deepcopy(q1)

        [q0_temp, q1_temp] = modify_qubits(
            zip(multi_qubits, [{"readout_range_out": 10}, {"readout_range_out": 20}]),
        )
        assert q0_temp.parameters.readout_range_out == 10
        assert q1_temp.parameters.readout_range_out == 20
        assert original_q0 == q0
        assert original_q1 == q1
        self._equal_except(q0, q0_temp, "readout_range_out")
        self._equal_except(q1, q1_temp, "readout_range_out")

        q0.parameters.nested_params = {"a": {"a": 2}, "b": 1}
        q1.parameters.nested_params = {"a": {"a": 3}, "b": 2}
        [q0_temp, q1_temp] = modify_qubits(
            zip(multi_qubits, [{"nested_params": {"a": 2}}, {"nested_params.a.a": 3}]),
        )
        assert q0_temp.parameters.nested_params == {"a": 2}
        assert q1_temp.parameters.nested_params == {"a": {"a": 3}, "b": 2}
        assert original_q0 == q0
        assert original_q1 == q1
        self._equal_except(q0, q0_temp, "nested_params")
        self._equal_except(q1, q1_temp, "nested_params")

        [q0_temp, q1_temp] = modify_qubits(
            [
                (q0, {"drive_parameters_ge.pulse.beta": 0.1}),
                (q1, {"drive_parameters_ef.pulse.sigma": 0.1}),
            ],
        )
        assert q0_temp.parameters.drive_parameters_ge["pulse"]["beta"] == 0.1
        assert q1_temp.parameters.drive_parameters_ef["pulse"]["sigma"] == 0.1
        assert original_q0 == q0
        assert original_q1 == q1
        self._equal_except(q0, q0_temp, "drive_parameters_ge")
        self._equal_except(q1, q1_temp, "drive_parameters_ef")

    def test_nonexisting_params_multiqubits(self, multi_qubits):
        q0, q1 = multi_qubits
        # test that updating non-existing parameters raises an error
        with pytest.raises(ValueError) as err:
            _ = modify_qubits(
                zip(
                    multi_qubits,
                    [
                        {
                            "non_existing_param": 10,
                            "readout_parameters.non_existing_param": 1,
                            "reset_delay_length": 1,
                        },
                        {"reset_delay_length": 10},
                    ],
                ),
            )
        assert str(err.value) == (f"Cannot update {q0.uid}")

        with pytest.raises(ValueError) as err:
            _ = modify_qubits(
                zip(
                    multi_qubits,
                    [
                        {"readout_range_out": 10},
                        {"non_existing_param": 1},
                    ],
                ),
            )

        assert str(err.value) == (f"Cannot update {q1.uid}")

    def test_temporary_qubits_context(self, multi_qubits):
        q0, q1 = multi_qubits
        with modify_qubits_context(
            zip(multi_qubits, [{"readout_range_out": 10}, {"readout_range_out": 20}]),
        ) as temp_qubits:
            q0_temp, q1_temp = temp_qubits
            assert q0_temp.parameters.readout_range_out == 10
            assert q1_temp.parameters.readout_range_out == 20
            assert q0 == q0_temp
            assert q1 == q1_temp
            self._equal_except(q0, q0_temp, "readout_range_out")
            self._equal_except(q1, q1_temp, "readout_range_out")


class TestTunableTransmonParameters:
    def test_create(self):
        p = TunableTransmonQubitParameters()

        assert p.readout_range_out == 5
        assert p.readout_range_in == 10
