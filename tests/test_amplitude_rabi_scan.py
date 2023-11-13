import pytest
from laboneq.simple import *  # noqa: F403

from laboneq_library.automatic_tuneup.tuneup.analyzer import *
from laboneq_library.automatic_tuneup.tuneup.experiment import *  # noqa: F403
from laboneq_library.automatic_tuneup.tuneup.scan import Scan

"""
Test concrete scan objects.
"""


@pytest.fixture(scope="function")
def amp_rabi_spec_scan(qubit_configs, session):
    amp_sweep = LinearSweepParameter(start=0.01, stop=1, count=110)
    Lorentzian()
    exp_settings = {"integration_time": 10e-6, "num_averages": 2**5}
    q0 = qubit_configs[0].qubit

    # Add extra attributes to qubit_configs
    q0.parameters.user_defined["pi_pulse_amplitude"] = 0.8
    qubit_configs[0].parameter.amplitude = [amp_sweep]

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
    qubit_configs[0].pulse = pulse_storage

    scan_amp_rabi = Scan(
        uid="amplitude_rabi_scan",
        session=session,
        qubit_configs=qubit_configs,
        exp_fac=AmplitudeRabi,
        exp_settings=exp_settings,
    )
    return scan_amp_rabi


def test_amp_rabi_spec_scan(amp_rabi_spec_scan, qubit_configs):
    qubit_configs[0].qubit
    amp_rabi_spec_scan.run(report=True, plot=False)
