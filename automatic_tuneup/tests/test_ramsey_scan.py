import pytest
from laboneq.simple import *  # noqa: F403
from tuneup.analyzer import RamseyAnalyzer
from tuneup.experiment import Ramsey
from tuneup.scan import Scan, ScanStatus

"""
Test concrete scan objects.
"""


@pytest.fixture(scope="function")
def ramsey_scan(session, qubits, set_bias_dc):
    delay_sweep = LinearSweepParameter(start=0, stop=10e-3, count=110)
    analyzer = RamseyAnalyzer()
    exp_settings = {"integration_time": 10e-6, "num_averages": 2**5}
    q0 = qubits[0]
    readout_pulse = pulse_library.const(uid="readout_pulse", length=2e-6, amplitude=0.9)
    drive_pulse = pulse_library.const(
        length=1e-7,
        amplitude=1,
    )
    pulse_storage = {"readout_pulse": readout_pulse, "drive_pulse": drive_pulse}
    ramsey_scan = Scan(
        uid="ramsey_scan",
        session=session,
        qubit=q0,
        params=[delay_sweep],
        update_key="pi_pulse_amplitude",
        exp_fac=Ramsey,
        exp_settings=exp_settings,
        analyzer=analyzer,
        pulse_storage=pulse_storage,
    )
    return ramsey_scan


def test_ramsey_scan(ramsey_scan):
    ramsey_scan.run(report=True, plot=False)
