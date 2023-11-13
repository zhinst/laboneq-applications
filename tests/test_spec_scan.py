import pytest
from laboneq.simple import *  # noqa: F403

from laboneq_library.automatic_tuneup.tuneup.analyzer import *  # noqa: F403
from laboneq_library.automatic_tuneup.tuneup.experiment import *  # noqa: F403
from laboneq_library.automatic_tuneup.tuneup.scan import Scan, ScanStatus

from .helpers import sim_resonances

"""
Test concrete scan objects.
"""


@pytest.fixture(scope="function")
def resonator_spec_scans(qubit_configs, session, set_bias_dc):
    # test class for scan
    freq_sweep = LinearSweepParameter(start=-300e6, stop=300e6, count=110)
    dc_sweep = LinearSweepParameter(start=-1, stop=1, count=20)
    session.register_user_function(set_bias_dc)
    spec_analyzer = Lorentzian(truth=1, tolerance=0.1, handles=["res_spec"])
    exp_settings = {"integration_time": 10e-6, "num_averages": 2**5}

    # Add/modify properties to qubit_configs for the following tests
    qubit_configs[0].qubit
    qubit_configs[0].parameter.frequency = [freq_sweep]
    qubit_configs[0].parameter.flux = [dc_sweep]
    qubit_configs[0].analyzer = spec_analyzer

    qubit_configs[1].need_to_verify = False

    scan_dc_sweep = Scan(
        uid="res_spec_dc_sweep",
        session=session,
        qubit_configs=qubit_configs,
        exp_fac=ReadoutSpectroscopyCWBiasSweep,
        exp_settings=exp_settings,
        ext_call=set_bias_dc,
    )

    scan = Scan(
        uid="res_spec_dc",
        session=session,
        qubit_configs=qubit_configs,
        exp_fac=ResonatorCWSpec,
        exp_settings=exp_settings,
        ext_call=set_bias_dc,
    )

    return scan, scan_dc_sweep


def test_resonator_spec_scans(resonator_spec_scans):
    scan, scan_dc_sweep = resonator_spec_scans
    scan.set_extra_calibration(measure_range=0, acquire_range=10)

    c = scan.experiment.get_calibration()
    assert c[scan.qubits[0].signals["measure"]].range == 0
    assert c[scan.qubits[0].signals["acquire"]].range == 10

    scan.run(report=True, plot=False)
    assert scan.status == ScanStatus.FINISHED

    f0 = 1
    scan.result = sim_resonances(f0=f0)

    # Update analyzer
    spec_analyzer = Lorentzian(truth=1, tolerance=0.1, handles=["res_spec"], f0=1, a=1)
    scan.qubit_configs[0].analyzer = spec_analyzer

    scan.analyze()
    scan.verify()

    assert scan.status == ScanStatus.PASSED

    scan.update()
    assert (
        scan.qubit_configs[0].qubit.parameters.readout_resonator_frequency
        == f0 + scan.qubit_configs[0].qubit.parameters.readout_lo_frequency
    )


def test_resonator_spec_scans_dc(resonator_spec_scans):
    _, scan = resonator_spec_scans
    scan.set_extra_calibration(measure_range=0, acquire_range=10)
    c = scan.experiment.get_calibration()
    assert c[scan.qubits[0].signals["measure"]].range == 0
    assert c[scan.qubits[0].signals["acquire"]].range == 10

    scan.run(report=True, plot=False)
    assert scan.status == ScanStatus.FINISHED


@pytest.fixture(scope="function")
def pulsed_resonator_spec_scan(session, qubit_configs):
    freq_sweep = LinearSweepParameter(start=-300e6, stop=300e6, count=110)
    spec_analyzer = Lorentzian()
    exp_settings = {"integration_time": 10e-6, "num_averages": 2**5}

    readout_pulse = pulse_library.const(uid="readout_pulse", length=2e-6, amplitude=0.9)
    pulse_storage = {"readout_pulse": readout_pulse}

    # Add/modify properties to qubit_configs for the following tests
    qubit_configs[0].qubit
    qubit_configs[0].parameter.frequency = [freq_sweep]
    qubit_configs[0].analyzer = spec_analyzer
    qubit_configs[0].pulses = pulse_storage

    scan = Scan(
        uid="pulsed_resonator_spec_scan",
        session=session,
        qubit_configs=qubit_configs,
        exp_fac=ResonatorPulsedSpec,
        exp_settings=exp_settings,
    )
    return scan


def test_pulsed_resonator_spec_scan(pulsed_resonator_spec_scan):
    scan = pulsed_resonator_spec_scan
    scan.run(report=True, plot=False)
    assert scan.status == ScanStatus.FINISHED


@pytest.fixture(scope="function")
def pulsed_qubit_spectroscopy(session, qubit_configs):
    freq_sweep = LinearSweepParameter(start=-300e6, stop=300e6, count=110)
    spec_analyzer = Lorentzian()
    exp_settings = {"integration_time": 10e-6, "num_averages": 2**5}

    readout_pulse = pulse_library.const(uid="readout_pulse", length=2e-6, amplitude=1.0)
    kernel_pulse = pulse_library.const(uid="kernel_pulse", length=2e-6, amplitude=1.0)
    drive_pulse = pulse_library.const(
        length=1e-7,
        amplitude=0.01,
    )
    pulse_storage = {
        "readout_pulse": readout_pulse,
        "drive_pulse": drive_pulse,
        "kernel_pulse": kernel_pulse,
    }

    qubit_configs[0].parameter.frequency = [freq_sweep]
    qubit_configs[0].analyzer = spec_analyzer
    qubit_configs[0].pulses = pulse_storage
    qubit_configs[1].need_to_verify = False
    qubit_configs[0].update_key = "resonance_frequency_ge"

    scan = Scan(
        uid="pulsed_qspec_spec_scan",
        session=session,
        qubit_configs=qubit_configs,
        exp_fac=PulsedQubitSpectroscopy,
        exp_settings=exp_settings,
    )
    return scan


def test_pulsed_qubit_spectroscopy(pulsed_qubit_spectroscopy):
    scan = pulsed_qubit_spectroscopy
    scan.run(report=True, plot=False)
    assert scan.status == ScanStatus.FINISHED

    f0 = 1
    scan.result = sim_resonances(f0=f0)

    # Update analyzer
    spec_analyzer = Lorentzian(truth=1, tolerance=0.1, handles=["res_spec"], f0=1, a=1)
    scan.qubit_configs[0].analyzer = spec_analyzer

    scan.analyze()
    scan.verify()

    assert scan.status == ScanStatus.PASSED
