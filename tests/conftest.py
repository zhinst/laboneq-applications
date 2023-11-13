import pytest
from laboneq.simple import *

from laboneq_library.automatic_tuneup.tuneup.analyzer import *
from laboneq_library.automatic_tuneup.tuneup.experiment import *
from laboneq_library.automatic_tuneup.tuneup.params import SweepParams
from laboneq_library.automatic_tuneup.tuneup.qubit_config import (
    QubitConfig,
    QubitConfigs,
)

from .helpers import generate_scans


@pytest.fixture(scope="module")
def device_setup():

    sherloq_descriptor = """\
    instruments:
      SHFQC:
        - address: dev12250
          uid: shfqc_psi
      SHFQA:
        - address: dev12251
          uid: shfqa
      PQSC:
        - address: dev10000
          uid: pqsc
    connections:
        shfqc_psi:
        - iq_signal: q0/drive_line
          ports: SGCHANNELS/0/OUTPUT
        - iq_signal: q0/measure_line
          ports: [QACHANNELS/0/OUTPUT]
        - acquire_signal: q0/acquire_line
          ports: [QACHANNELS/0/INPUT]

        - iq_signal: q1/drive_line
          ports: SGCHANNELS/1/OUTPUT

        shfqa:
        - iq_signal: q1/measure_line
          ports: [QACHANNELS/0/OUTPUT]
        - acquire_signal: q1/acquire_line
          ports: [QACHANNELS/0/INPUT]

        pqsc:
        - to: shfqc_psi
          ports: ZSYNCS/0
        - to: shfqa
          ports: ZSYNCS/1

    """

    device_setup = DeviceSetup.from_descriptor(
        sherloq_descriptor,
        server_host="localhost",
        server_port=8004,
        setup_name="sherloq",
    )

    my_calibration = Calibration()
    my_calibration["/logical_signal_groups/q0/drive_line"] = SignalCalibration(
        oscillator=Oscillator(frequency=1.23e6),
    )
    my_calibration["/logical_signal_groups/q1/drive_line"] = SignalCalibration(
        oscillator=Oscillator(frequency=3.21e6),
    )
    my_calibration["/logical_signal_groups/q0/measure_line"] = SignalCalibration(
        oscillator=Oscillator(frequency=1.23e6),
        local_oscillator=Oscillator(frequency=5e9),
    )
    my_calibration["/logical_signal_groups/q1/measure_line"] = SignalCalibration(
        oscillator=Oscillator(frequency=1.23e6),
        local_oscillator=Oscillator(frequency=5e9),
    )

    device_setup.set_calibration(my_calibration)

    return device_setup


@pytest.fixture(scope="function")
def qubits(device_setup):
    q0 = Transmon.from_logical_signal_group(
        "q0",
        lsg=device_setup.logical_signal_groups["q0"],
        parameters=TransmonParameters(
            resonance_frequency_ge=6.1e9,
            resonance_frequency_ef=7e9,
            drive_lo_frequency=6e9,
            readout_resonator_frequency=3.8e9,
            readout_lo_frequency=3.7e9,
            user_defined={
                "readout_len": 15e-6,
                "readout_amp": 1.0,
                "pi_pulse_amplitude": 1.0,
                "test_key": 1.0,
            },
        ),
    )

    q1 = Transmon.from_logical_signal_group(
        "q1",
        lsg=device_setup.logical_signal_groups["q1"],
        parameters=TransmonParameters(
            resonance_frequency_ge=6.1e9,
            resonance_frequency_ef=7e9,
            drive_lo_frequency=6e9,
            readout_resonator_frequency=3.8e9,
            readout_lo_frequency=3.7e9,
            user_defined={
                "readout_len": 15e-6,
                "readout_amp": 1.0,
                "pi_pulse_amplitude": 1.0,
                "test_key": 1.0,
            },
        ),
    )
    return (q0, q1)


@pytest.fixture(scope="module")
def session(device_setup):
    my_session = Session(device_setup=device_setup)
    my_session.connect(do_emulation=True)
    return my_session


@pytest.fixture(scope="function")
def qubit_configs(qubits):
    q0, q1 = qubits
    sweep1 = LinearSweepParameter(start=-300e6, stop=300e6, count=11)
    sweep2 = LinearSweepParameter(start=-300e6, stop=300e6, count=11)

    param = SweepParams(frequency=sweep1)
    param2 = SweepParams(frequency=sweep2)

    qconfig0 = QubitConfig(param, q0, update_key="readout_resonator_frequency")
    qconfig1 = QubitConfig(param2, q1, update_key="readout_resonator_frequency")
    qconfigs = QubitConfigs([qconfig0, qconfig1])

    return qconfigs


@pytest.fixture(scope="function")
def set_bias_dc():
    def set_bias_dc(session, qubit_uid, voltage):
        print(f"Setting bias dc to {voltage} V")

    return set_bias_dc


@pytest.fixture(scope="function")
def scan_single_qubit(session, qubits):
    return generate_scans(1, session, qubits)[0]
