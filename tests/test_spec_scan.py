import pytest
from laboneq.simple import *  # noqa: F403

from laboneq_library.automatic_tuneup.tuneup.analyzer import *  # noqa: F403
from laboneq_library.automatic_tuneup.tuneup.experiment import *  # noqa: F403
from laboneq_library.automatic_tuneup.tuneup.scan import Scan, ScanStatus

"""
Test concrete scan objects.
"""


@pytest.fixture(scope="function")
def resonator_spec_scans(session, qubits, set_bias_dc):
    # test class for scan
    freq_sweep = LinearSweepParameter(start=-300e6, stop=300e6, count=110)
    dc_sweep = LinearSweepParameter(start=-1, stop=1, count=20)
    session.register_user_function(set_bias_dc)
    spec_analyzer = ResonatorSpectAnalyzerTranx()
    exp_settings = {"integration_time": 10e-6, "num_averages": 2**5}
    q0 = qubits[0]
    scan_dc_sweep = Scan(
        uid="res_spec_dc_sweep",
        session=session,
        qubit=q0,
        params=[freq_sweep, dc_sweep],
        update_key="readout_resonator_frequency",
        exp_fac=ReadoutSpectroscopyCWBiasSweep,
        exp_settings=exp_settings,
        analyzer=spec_analyzer,
        ext_call=set_bias_dc,
    )

    scan = Scan(
        uid="res_spec_dc",
        session=session,
        qubit=q0,
        params=[freq_sweep],
        update_key="readout_resonator_frequency",
        exp_fac=RSpecCwFactory,
        exp_settings=exp_settings,
        analyzer=spec_analyzer,
        ext_call=set_bias_dc,
    )

    return scan, scan_dc_sweep


def test_resonator_spec_scans(resonator_spec_scans):
    scan, scan_dc_sweep = resonator_spec_scans
    scan.set_extra_calibration(measure_range=0, acquire_range=10)
    c = scan.experiment.get_calibration()
    assert c[scan.qubit.signals["measure"]].range == 0
    assert c[scan.qubit.signals["acquire"]].range == 10

    scan.run(report=True, plot=False)
    assert scan.status == ScanStatus.FINISHED

    scan_dc_sweep.run(report=True, plot=False)
    assert scan_dc_sweep.status == ScanStatus.FINISHED


@pytest.fixture(scope="function")
def pulsed_resonator_spec_scan(session, qubits, set_bias_dc):
    # test class for scan
    freq_sweep = LinearSweepParameter(start=-300e6, stop=300e6, count=110)
    session.register_user_function(set_bias_dc)
    spec_analyzer = ResonatorSpectAnalyzerTranx()
    exp_settings = {"integration_time": 10e-6, "num_averages": 2**5}
    q0 = qubits[0]
    readout_pulse = pulse_library.const(uid="readout_pulse", length=2e-6, amplitude=0.9)
    pulse_storage = {"readout_pulse": readout_pulse}
    scan = Scan(
        uid="pulsed_resonator_spec_scan",
        session=session,
        qubit=q0,
        params=[freq_sweep],
        update_key="readout_resonator_frequency",
        exp_fac=ReadoutSpectroscopyPulsed,
        exp_settings=exp_settings,
        analyzer=spec_analyzer,
        ext_call=set_bias_dc,
        pulse_storage=pulse_storage,
    )
    return scan


def test_pulsed_resonator_spec_scan(pulsed_resonator_spec_scan):
    scan = pulsed_resonator_spec_scan
    scan.run(report=True, plot=False)


@pytest.fixture(scope="function")
def pulsed_qubit_spectroscopy(session, qubits):
    freq_sweep = LinearSweepParameter(start=-300e6, stop=300e6, count=110)
    spec_analyzer = ResonatorSpectAnalyzerTranx()
    exp_settings = {"integration_time": 10e-6, "num_averages": 2**5}
    q0 = qubits[0]
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
    scan = Scan(
        uid="pulsed_qspec_spec_scan",
        session=session,
        qubit=q0,
        params=[freq_sweep],
        update_key="resonance_frequency_ge",
        exp_fac=PulsedQubitSpectroscopy,
        exp_settings=exp_settings,
        analyzer=spec_analyzer,
        pulse_storage=pulse_storage,
    )
    return scan


def test_pulsed_qubit_spectroscopy(pulsed_qubit_spectroscopy):
    scan = pulsed_qubit_spectroscopy
    scan.run(report=True, plot=False)
