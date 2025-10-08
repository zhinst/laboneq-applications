# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Tests for laboneq_applications.qpu_types.tunable_coupler.qpu_types."""

import copy

import pytest

from laboneq_applications.qpu_types.tunable_coupler import (
    TunableCoupler,
    TunableCouplerParameters,
)


@pytest.fixture
def c0():
    return TunableCoupler(uid="c0", signals={"flux": "c0/flux"})


class TestTunableCoupler:
    def test_create(self):
        signals = {
            "flux": "c0/flux",
        }
        c = TunableCoupler(uid="c0", signals=signals)
        assert c.uid == "c0"
        assert isinstance(c.parameters, TunableCouplerParameters)

    def test_gate_parameters(self, c0):
        assert c0.gate_parameters("iswap") == (
            "flux",
            {
                "pulse": {"function": "gaussian_square"},
                "amplitude": None,
                "length": None,
                "frequency": None,
            },
        )

    def test_set_custom_parameters(self, c0):
        assert c0.parameters.custom == {}

        c0.parameters.custom["my_cool_parameter"] = 1.0
        assert c0.parameters.custom == {"my_cool_parameter": 1.0}

    def test_update(self, c0):
        assert (
            c0.parameters.gate_parameters["iswap"]["pulse"]["function"]
            == "gaussian_square"
        )
        c0.update(
            gate_parameters={
                "iswap": {
                    "pulse": {"function": "const"},
                    "amplitude": None,
                    "length": None,
                    "frequency": None,
                }
            }
        )
        assert c0.parameters.gate_parameters["iswap"]["pulse"]["function"] == "const"

        c0.update(dc_slot=2)
        assert c0.parameters.dc_slot == 2

        c0.update(dc_voltage_parking=10e-6)
        assert c0.parameters.dc_voltage_parking == 10e-6

        c0.update(flux_offset_voltage=10e-6)
        assert c0.parameters.flux_offset_voltage == 10e-6

    def test_update_nonexisting_params(self, c0):
        # test that updating non-existing parameters raises an error

        original_params = copy.deepcopy(c0.parameters)
        with pytest.raises(ValueError) as err:
            c0.update(
                **{
                    "dc_slot": 10,
                    "non_existing_param": 10,
                    "gate_parameters.non_existing_param": 10,
                },
            )

        assert str(err.value) == (
            "Update parameters do not match the qubit parameters:"
            " ['non_existing_param', 'gate_parameters.non_existing_param']"
        )
        # assert no parameters were updated
        assert c0.parameters == original_params

    def test_replace(self, c0):
        new_c0 = c0.replace(
            dc_slot=1,
            dc_voltage_parking=10e-6,
            flux_offset_voltage=0.1,
        )
        assert id(new_c0) != id(c0)
        assert new_c0.parameters.dc_slot == 1
        assert new_c0.parameters.dc_voltage_parking == 10e-6
        assert new_c0.parameters.flux_offset_voltage == 0.1

    def test_replace_wrong_params(self, c0):
        with pytest.raises(ValueError) as exc_info:
            _ = c0.replace(
                wrong_param=0,
                wrong_param_2=1,
            )
        assert str(exc_info.value) == (
            "Update parameters do not match the qubit parameters:"
            " ['wrong_param', 'wrong_param_2']"
        )

    def test_invalid_params_reported_correctly(self, c0):
        non_existing_params = [
            "non_existing_param",
            "gate_parameters.non_existing_param",
        ]
        with pytest.raises(ValueError) as err:
            c0.parameters.replace(
                **{
                    "dc_slot": 10,
                    "non_existing_param": 10,
                    "gate_parameters.non_existing_param": 10,
                },
            )

        assert str(err.value) == (
            f"Update parameters do not match the qubit "
            f"parameters: {non_existing_params}"
        )

        # nested invalid parameters are reported correctly
        non_existing_params = [
            "gate_parameters.non_existing.not_existing",
            "non_existing.not_existing",
        ]
        with pytest.raises(ValueError) as err:
            c0.parameters.replace(
                **{
                    "gate_parameters.non_existing.not_existing": 10,
                    "non_existing.not_existing": 10,
                },
            )

        assert str(err.value) == (
            f"Update parameters do not match the qubit "
            f"parameters: {non_existing_params}"
        )

    def test_calibration(self, c0):
        c0.parameters.flux_offset_voltage = 1.23

        coupler_calib = c0.calibration()
        acq_sig_calib = coupler_calib[c0.signals["flux"]]

        assert acq_sig_calib.voltage_offset == 1.23


class TestTunableCouplerParameters:
    def test_create(self):
        p = TunableCouplerParameters()

        assert p.gate_parameters == {
            "iswap": {
                "pulse": {"function": "gaussian_square"},
                "amplitude": None,
                "length": None,
                "frequency": None,
            }
        }
        assert p.dc_slot == 0
        assert p.dc_voltage_parking is None
        assert p.flux_offset_voltage == 0.0

    def test_gate_parameters(self):
        p1 = TunableCouplerParameters()
        p2 = TunableCouplerParameters()

        assert p1.gate_parameters["iswap"]["amplitude"] is None
        assert p2.gate_parameters["iswap"]["amplitude"] is None
        p1.gate_parameters["iswap"]["amplitude"] = 1
        assert p1.gate_parameters["iswap"]["amplitude"] == 1
        assert p2.gate_parameters["iswap"]["amplitude"] is None

    def test_gate_parameters_error(self):
        p = TunableCouplerParameters()

        with pytest.raises(KeyError):
            p.gate_parameters["unknown_key"]

    def test_dc_slot(self):
        p = TunableCouplerParameters(dc_slot=5)

        assert p.dc_slot == 5

    def test_dc_voltage_parking(self):
        p = TunableCouplerParameters(dc_voltage_parking=5.0)

        assert p.dc_voltage_parking == 5.0

    def test_flux_offset_voltage(self):
        p = TunableCouplerParameters(flux_offset_voltage=5.0)

        assert p.flux_offset_voltage == 5.0

    def test_custom(self):
        p = TunableCouplerParameters()

        assert p.custom == {}
