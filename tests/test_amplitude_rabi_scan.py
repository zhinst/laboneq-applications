import pytest
from laboneq.simple import *  # noqa: F403

from laboneq_library.automatic_tuneup.tuneup.analyzer import *
from laboneq_library.automatic_tuneup.tuneup.experiment import *  # noqa: F403
from laboneq_library.automatic_tuneup.tuneup.scan import Scan

"""
Test concrete scan objects.
"""


@pytest.fixture(scope="function")
def amp_rabi_spec_scan(session, qubits, set_bias_dc):
    # test class for scan
    amp_sweep = LinearSweepParameter(start=0.01, stop=1, count=110)
    spec_analyzer = Lorentzian()
    exp_settings = {"integration_time": 10e-6, "num_averages": 2**5}
    q0 = qubits[0]
    q0.parameters.user_defined["pi_pulse_amplitude"] = 0.8
    readout_pulse = pulse_library.const(uid="readout_pulse", length=2e-6, amplitude=1.0)
    kernel_pulse = pulse_library.const(uid="kernel_pulse", length=2e-6, amplitude=1.0)
    drive_pulse = pulse_library.const(
        length=1e-7,
        amplitude=1,
    )
    pulse_storage = {
        "readout_pulse": readout_pulse,
        "drive_pulse": drive_pulse,
        "kernel_pulse": kernel_pulse,
    }
    scan_amp_rabi = Scan(
        uid="amplitude_rabi_scan",
        session=session,
        qubit=q0,
        params=[amp_sweep],
        update_key="pi_pulse_amplitude",
        exp_fac=AmplitudeRabi,
        exp_settings=exp_settings,
        analyzer=spec_analyzer,
        ext_call=set_bias_dc,
        pulse_storage=pulse_storage,
    )
    return scan_amp_rabi


def test_amp_rabi_spec_scan(amp_rabi_spec_scan, qubits):
    q0 = qubits[0]
    amp_rabi_spec_scan.run(report=True, plot=False)
    assert (
        amp_rabi_spec_scan.pulse_storage["drive_pulse"].amplitude
        == q0.parameters.user_defined["pi_pulse_amplitude"]
    )
