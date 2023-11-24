from laboneq import *

import laboneq_library.automatic_tuneup.tuneup.analyzer as ta
from laboneq_library.automatic_tuneup.tuneup import TuneUp
from laboneq_library.automatic_tuneup.tuneup.experiment import *
from laboneq_library.automatic_tuneup.tuneup.helper import (
    get_device_setup_qzilla_shfqc,
    get_device_setup_sherloq,
    save_qubits,
)
from laboneq_library.automatic_tuneup.tuneup.scan import *
from laboneq_library.automatic_tuneup.tuneup.qubit_config import QubitConfig, QubitConfigs
from laboneq_library.automatic_tuneup.tuneup.params import SweepParams

EMULATION = True
TEST_ON_SCOPE = False
if TEST_ON_SCOPE:
    device_setup = get_device_setup_qzilla_shfqc()
else:
    device_setup = get_device_setup_sherloq()
session = Session(device_setup=device_setup)
session.connect(do_emulation=EMULATION)

mock_session = Session(device_setup=device_setup)
mock_session.connect(do_emulation=True)

# define qubits
q0 = Transmon.from_logical_signal_group(
    "q0",
    lsg=device_setup.logical_signal_groups["q0"],
    parameters=TransmonParameters(
        resonance_frequency_ge=4.7e9,
        resonance_frequency_ef=7e9,
        drive_lo_frequency=4.7e9,
        readout_resonator_frequency=7e9,
        readout_lo_frequency=7e9,
        drive_range=5,
        readout_range_out=-30,
        readout_range_in=-5,
        user_defined={"pi_pulse_amplitude": 1.0},
    ),
)

q1 = Transmon.from_logical_signal_group(
    "q1",
    lsg=device_setup.logical_signal_groups["q1"],
    parameters=TransmonParameters(
        resonance_frequency_ge=6.1e9,
        resonance_frequency_ef=3906609992.405414,
        drive_lo_frequency=4e9,
        readout_resonator_frequency=0e6,
        readout_lo_frequency=7e9,
        drive_range=5,
        readout_range_out=-30,
        readout_range_in=-5,
        user_defined={"pi_pulse_amplitude": 1.0},
    ),
)


# setting up scans
def generate_pulsed_resonator_scan():
    freq_sweep = LinearSweepParameter(start=35e6, stop=45e6, count=210)
    spec_analyzer = ta.MockAnalyzer(handles=["res_spec"])
    exp_settings = {"integration_time": 10e-6, "num_averages": 2**10}
    readout_pulse = pulse_library.const(
        uid="readout_pulse", length=2e-6, amplitude=0.05
    )
    kernel_pulse = pulse_library.const(uid="kernel_pulse", length=2e-6, amplitude=1.0)
    pulse_storage = {"readout_pulse": readout_pulse, "kernel_pulse": kernel_pulse}

    param0 = SweepParams(frequency=freq_sweep)
    qconfig0 = QubitConfig(param0,q0,update_key="readout_resonator_frequency", pulses=pulse_storage, analyzer=spec_analyzer)
    qconfigs = QubitConfigs([qconfig0])

    scan_prs = Scan(
        uid="pulsed_resonator_spec",
        session=session,
        qubit_configs=qconfigs,
        exp_fac=ResonatorPulsedSpec,
        exp_settings=exp_settings,
    )

    scan_prs.set_extra_calibration(measure_range=-30)
    return scan_prs


def generate_qubit_spectroscopy_scan():
    freq_sweep = LinearSweepParameter(start=16e6, stop=22e6, count=201)
    spec_analyzer = ta.MockAnalyzer()
    exp_settings = {"num_averages": 2**11}
    readout_pulse = pulse_library.const(
        uid="readout_pulse", length=2e-6, amplitude=0.05
    )
    kernel_pulse = pulse_library.const(uid="kernel_pulse", length=2e-6, amplitude=1.0)
    drive_pulse = pulse_library.const(
        length=2.5e-5,
        amplitude=0.05,
    )
    pulse_storage = {
        "readout_pulse": readout_pulse,
        "drive_pulse": drive_pulse,
        "kernel_pulse": kernel_pulse,
    }

    param0 = SweepParams(frequency=freq_sweep)
    qconfig0 = QubitConfig(param0,q0,update_key="readout_resonator_frequency", pulses=pulse_storage, analyzer=spec_analyzer)
    qconfigs = QubitConfigs([qconfig0])

    scan_qspec = Scan(
        uid="pulsed_qspec",
        session=session,
        qubit_configs=qconfigs,
        exp_fac=PulsedQubitSpectroscopy,
        exp_settings=exp_settings,
    )
    scan_qspec.set_extra_calibration(drive_range=-25)
    return scan_qspec


def generate_rabi_scan():
    amp_sweep = LinearSweepParameter(start=0.01, stop=1, count=110)
    exp_settings = {"num_averages": 2**12}
    readout_pulse = pulse_library.const(
        uid="readout_pulse", length=2e-6, amplitude=0.05
    )
    kernel_pulse = pulse_library.const(uid="kernel_pulse", length=2e-6, amplitude=1.0)
    drive_pulse = pulse_library.gaussian(
        length=1e-7,
        amplitude=1,
    )
    pulse_storage = {
        "readout_pulse": readout_pulse,
        "drive_pulse": drive_pulse,
        "kernel_pulse": kernel_pulse,
    }
    rabi_analyzer = ta.RabiAnalyzer()

    param0 = SweepParams(amplitude=amp_sweep)
    qconfig0 = QubitConfig(param0,q0,update_key="pi_pulse_amplitude", pulses=pulse_storage, analyzer=rabi_analyzer)
    qconfigs = QubitConfigs([qconfig0])

    scan_amp_rabi = Scan(
        uid="amplitude_rabi",
        session=session,
        qubit_configs=qconfigs,
        exp_fac=AmplitudeRabi,
        exp_settings=exp_settings,
    )
    return scan_amp_rabi


def generate_mock_scan(uid):
    analyzer = ta.MockAnalyzer()
    qconfig0 = QubitConfig(None, q0, update_key="pi_pulse_amplitude", analyzer=analyzer)
    qconfigs = QubitConfigs([qconfig0])
    scan = Scan(uid=uid, session=mock_session, exp_fac=MockExp, qubit_configs=qconfigs)
    return scan


if __name__ == "__main__":
    scan_rabi = generate_rabi_scan()
    scan_q_spec = generate_qubit_spectroscopy_scan()
    scan_res_spec = generate_pulsed_resonator_scan()

    # mock
    scan_cw_rspec_bias_sweep = generate_mock_scan("cw_rspec_bias_sweep")
    scan_cw_rspec_bias_sweep_power = generate_mock_scan("cw_rspec_bias_sweep_power")
    scan_not_supposed_to_run = generate_mock_scan("not running")

    scan_rspec_power = generate_mock_scan("pulsed_resonator_power_sweep")

    scan_rabi.add_dependencies(scan_q_spec)
    scan_q_spec.add_dependencies(scan_res_spec)
    scan_res_spec.add_dependencies(
        [scan_cw_rspec_bias_sweep, scan_cw_rspec_bias_sweep_power]
    )

    # for demo
    scan_q_spec.add_dependencies(scan_rspec_power)

    # demonstrate stopped at failure
    # scan_rspec_power.analyzer = ta.AlwaysFailedAnalyzer()

    update_params = not TEST_ON_SCOPE
    verify = not TEST_ON_SCOPE
    analyze = not TEST_ON_SCOPE

    # Define tuneup
    tuneup = TuneUp(uid="tuneupPSI", scans=[scan_rabi])
    tuneup.run(
        scan_q_spec,
        plot_graph=True,
        stop_at_failed=True,
        analyze=analyze,
        verify=verify,
        update=update_params,
    )

    save_qubits(q0)
