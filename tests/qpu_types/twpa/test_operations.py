# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Tests for laboneq_applications.qpu_types.twpa.operations."""

import pytest
from laboneq.dsl.calibration import CancellationSource, Oscillator, SignalCalibration
from laboneq.dsl.device import DeviceSetup, create_connection
from laboneq.dsl.device.instruments import SHFQC
from laboneq.dsl.enums import ModulationType
from laboneq.simple import SectionAlignment, dsl

from laboneq_applications.qpu_types.twpa.operations import TWPAOperations
from laboneq_applications.qpu_types.twpa.twpa_types import TWPA, TWPAParameters

import tests.helpers.dsl as tsl


def twpa_setup() -> DeviceSetup:
    """Return a demo setup with a TWPA.

    The returned setup consists of 1 SHFQC (for the qubit drive and measurement lines)

    Returns:
        The device setup.
    """
    setup = DeviceSetup("twpa_setup")
    setup.add_dataserver(host="localhost", port="8004")

    setup.add_instruments(
        SHFQC(uid="device_shfqc", address="dev1234", device_options="SHFQC/QC6CH"),
    )

    setup.add_connections(
        "device_shfqc",
        create_connection(
            to_signal="q0/measure",
            ports="QACHANNELS/0/OUTPUT",
        ),
        create_connection(to_signal="q0/acquire", ports="QACHANNELS/0/INPUT"),
    )

    logical_signal = setup.logical_signal_by_uid("q0/measure")
    oscillator = Oscillator(modulation_type=ModulationType.AUTO)
    logical_signal.calibration = SignalCalibration(
        local_oscillator=Oscillator(frequency=4e9),
        oscillator=oscillator,
    )

    acquire_signal = setup.logical_signal_by_uid("q0/acquire")
    acquire_signal.calibration = SignalCalibration(
        local_oscillator=Oscillator(frequency=4e9),
        oscillator=oscillator,
    )

    return setup


@pytest.fixture
def twpa() -> TWPA:
    setup = twpa_setup()
    return TWPA.from_logical_signal_group(
        "twpa0",
        setup.logical_signal_groups["q0"],
        parameters=TWPAParameters(readout_lo_frequency=6.4e9, probe_frequency=6.4e9),
    )


@pytest.fixture
def twpa_op() -> TWPAOperations:
    return TWPAOperations()


class TestTWPAOperations:
    def test_create(self):
        qop = TWPAOperations()
        assert qop.QUBIT_TYPES is TWPA

    @pytest.mark.parametrize(
        ("rf", "freq", "oscillator_freq"),
        [
            pytest.param(True, 6.5e9, 0.1e9, id="rf-positive"),
            pytest.param(True, 6.3e9, -0.1e9, id="rf-negative"),
            pytest.param(False, 0.1e9, 0.1e9, id="oscillator-positive"),
            pytest.param(False, -0.1e9, -0.1e9, id="oscillator-negative"),
        ],
    )
    def test_set_readout_frequency(self, twpa, twpa_op, rf, freq, oscillator_freq):
        @dsl.qubit_experiment
        def exp_set_freq(q):
            with dsl.acquire_loop_rt(count=10):
                twpa_op.set_readout_frequency(q, freq, rf=rf)

        exp = exp_set_freq(twpa)
        calibration = exp.get_calibration()
        signal_calibration = calibration[twpa.signals["measure"]]
        assert signal_calibration.oscillator.frequency == oscillator_freq

    def test_set_readout_amplitude(self, twpa, twpa_op):
        @dsl.qubit_experiment
        def exp_set_freq(q):
            with dsl.acquire_loop_rt(count=10):
                twpa_op.set_readout_amplitude(q, 0.6)

        exp = exp_set_freq(twpa)
        calibration = exp.get_calibration()
        signal_calibration = calibration[twpa.signals["measure"]]
        assert signal_calibration.amplitude == 0.6

    def test_set_pump_frequency(self, twpa, twpa_op):
        @dsl.qubit_experiment
        def exp_set_freq(q):
            with dsl.acquire_loop_rt(count=10):
                twpa_op.set_pump_frequency(q, 5.1e9)

        exp = exp_set_freq(twpa)
        calibration = exp.get_calibration()
        signal_calibration = calibration[twpa.signals["acquire"]]
        assert signal_calibration.amplifier_pump.pump_frequency == 5.1e9

    def test_set_pump_power(self, twpa, twpa_op):
        @dsl.qubit_experiment
        def exp_set_freq(q):
            with dsl.acquire_loop_rt(count=10):
                twpa_op.set_pump_power(q, 1.1)

        exp = exp_set_freq(twpa)
        calibration = exp.get_calibration()
        signal_calibration = calibration[twpa.signals["acquire"]]
        assert signal_calibration.amplifier_pump.pump_power == 1.1

    @pytest.mark.parametrize(
        ("cancellation", "cancellation_source"),
        [
            pytest.param(True, CancellationSource.INTERNAL, id="cancellation-on"),
            pytest.param(False, CancellationSource.EXTERNAL, id="cancellation-off"),
        ],
    )
    def test_set_pump_cancellation(
        self, twpa, twpa_op, cancellation, cancellation_source
    ):
        @dsl.qubit_experiment
        def exp_set_freq(q):
            with dsl.acquire_loop_rt(count=10):
                twpa_op.set_pump_cancellation(
                    q,
                    cancellation_attenuation=1.0,
                    cancellation_phaseshift=0.0,
                    cancellation=cancellation,
                )

        exp = exp_set_freq(twpa)
        calibration = exp.get_calibration()
        signal_calibration = calibration[twpa.signals["acquire"]]
        assert signal_calibration.amplifier_pump.cancellation_phase == 0.0
        assert signal_calibration.amplifier_pump.cancellation_attenuation == 1.0
        assert (
            signal_calibration.amplifier_pump.cancellation_source == cancellation_source
        )

    def test_twpa_measure(self, twpa, twpa_op):
        measure_section = twpa_op.twpa_measure(twpa, handle="twpa_measure")
        assert measure_section == tsl.section(
            uid="__twpa_measure_twpa0_0",
            alignment=SectionAlignment.LEFT,
        ).children(
            tsl.reserve_op(signal="q0/measure"),
            tsl.reserve_op(signal="q0/acquire"),
            tsl.play_pulse_op(
                signal="q0/measure",
                amplitude=0.5,
                length=5e-6,
                increment_oscillator_phase=None,
                phase=None,
                pulse_parameters=None,
                pulse=tsl.pulse(
                    uid="__readout_pulse_0",
                    function="const",
                    amplitude=1.0,
                    length=1e-7,
                    pulse_parameters=None,
                ),
            ),
            tsl.acquire_op(
                signal="q0/acquire",
                handle="twpa_measure",
                kernel=None,
                length=5e-6,
                pulse_parameters=None,
            ),
        )

    def test_twpa_acquire(self, twpa, twpa_op):
        acquire_section = twpa_op.twpa_acquire(twpa, handle="acquire_twpa")
        assert acquire_section == tsl.section(
            uid="__twpa_acquire_twpa0_0",
            alignment=SectionAlignment.LEFT,
        ).children(
            tsl.reserve_op(signal="q0/measure"),
            tsl.reserve_op(signal="q0/acquire"),
            tsl.acquire_op(
                signal="q0/acquire",
                handle="acquire_twpa",
                kernel=None,
                length=5e-6,
                pulse_parameters=None,
            ),
        )
