import pytest
from laboneq.simple import *  # noqa: F403

from laboneq_library.automatic_tuneup.tuneup.analyzer import RamseyAnalyzer
from laboneq_library.automatic_tuneup.tuneup.experiment import Ramsey
from laboneq_library.automatic_tuneup.tuneup.scan import Scan

"""
Test concrete scan objects.
"""


@pytest.fixture(scope="function")
def ramsey_scan(qubit_configs, session):
    delay_sweep = LinearSweepParameter(start=0, stop=10e-3, count=110)
    RamseyAnalyzer()
    exp_settings = {"integration_time": 10e-6, "num_averages": 2**5}
    q0 = qubit_configs[0].qubit

    readout_pulse = pulse_library.const(uid="readout_pulse", length=2e-6, amplitude=0.9)
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

    # Add extra attributes to qubit_configs
    q0.parameters.user_defined["pi_pulse_amplitude"] = 0.8
    qubit_configs[0].parameter.delay = [delay_sweep]
    qubit_configs[0].pulse = pulse_storage

    ramsey_scan = Scan(
        uid="ramsey_scan",
        session=session,
        qubit_configs=qubit_configs,
        exp_fac=Ramsey,
        exp_settings=exp_settings,
    )
    return ramsey_scan


def test_ramsey_scan(ramsey_scan):
    ramsey_scan.run(report=True, plot=False)
