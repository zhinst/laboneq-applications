from laboneq.simple import *
from laboneq._utils import id_generator

# additional imports
import numpy as np

# TODO: Add port delay and single shot readout experiments


# convenience
def flatten(l):
    # flatten a nested list to a single level
    return [item for sublist in l for item in sublist]


# pulses
def qubit_gaussian_pulse(qubit):
    return pulse_library.gaussian(
        uid=f"gaussian_pulse_drive_{qubit.uid}",
        length=qubit.parameters.user_defined["pulse_length"],
        amplitude=1.0,
        sigma=0.3,
    )


def qubit_drive_pulse(qubit, amplitude=None):
    if amplitude is None:
        amplitude = qubit.parameters.user_defined["amplitude_pi"]
    return pulse_library.drag(
        uid=f"drag_pulse_{qubit.uid}",
        length=qubit.parameters.user_defined["pulse_length"],
        amplitude=amplitude,
        sigma=0.3,
        beta=0.0,
    )


def create_amp_sweep(id, start_amp, stop_amp, num_points):
    return LinearSweepParameter(
        uid=f"amp_sweep_{id}",
        start=start_amp,
        stop=stop_amp,
        count=num_points,
    )


def readout_gauss_square_pulse(qubit):
    return pulse_library.gaussian_square(
        uid=f"readout_pulse_{qubit.uid}",
        length=qubit.parameters.user_defined["readout_length"],
        amplitude=qubit.parameters.user_defined["readout_amplitude"],
    )


def qubit_spectroscopy_pulse(qubit):
    return pulse_library.const(
        uid=f"spectroscopy_pulse_{qubit.uid}",
        length=qubit.parameters.user_defined["readout_length"],
        amplitude=qubit.parameters.user_defined["readout_amplitude"],
        # can_compress=True,
    )


def qubit_gaussian_pulse(qubit):
    return pulse_library.gaussian(
        uid=f"gaussian_pulse_drive_{qubit.uid}",
        length=qubit.parameters.user_defined["pulse_length"],
        amplitude=qubit.parameters.user_defined["amplitude_pi"],
    )


def qubit_gaussian_halfpi_pulse(qubit):
    return pulse_library.gaussian(
        uid=f"gaussian_pulse_drive_{qubit.uid}",
        length=qubit.parameters.user_defined["pulse_length"],
        amplitude=qubit.parameters.user_defined["amplitude_pi2"],
    )


def readout_pulse(qubit):
    return pulse_library.const(
        uid=f"readout_pulse_{qubit.uid}",
        length=qubit.parameters.user_defined["readout_length"],
        amplitude=qubit.parameters.user_defined["readout_amplitude"],
    )


def integration_kernel(qubit):
    return pulse_library.const(
        uid=f"integration_kernel_{qubit.uid}",
        length=qubit.parameters.user_defined["readout_length"],
        amplitude=1,
    )


# define sweep parameter
def create_freq_sweep(
    id, start_freq, stop_freq, num_points, axis_name="Frequency [Hz]"
):
    return LinearSweepParameter(
        uid=f"frequency_sweep_{id}",
        start=start_freq,
        stop=stop_freq,
        count=num_points,
        axis_name=axis_name,
    )


def resonator_spectroscopy_parallel_CW_full_range(
    qubits,
    outer_sweep,
    inner_sweep,
    measure_range=-25,
    acquire_range=-25,
    integration_time=10e-3,
    num_averages=1,
):
    # Create resonator spectroscopy experiment - uses only readout drive and signal acquisition
    exp_spec = Experiment(
        uid="Resonator Spectroscopy CW",
        signals=[
            signal
            for signal_list in [
                [
                    ExperimentSignal(
                        f"measure_{qubit.uid}", map_to=qubit.signals["measure"]
                    ),
                    ExperimentSignal(
                        f"acquire_{qubit.uid}", map_to=qubit.signals["acquire"]
                    ),
                ]
                for qubit in qubits
            ]
            for signal in signal_list
        ],
    )

    ## define experimental sequence
    # loop - average multiple measurements for each frequency - measurement in spectroscopy mode
    with exp_spec.sweep(uid="resonator_frequency_outer", parameter=outer_sweep):
        with exp_spec.acquire_loop_rt(
            uid="shots",
            count=num_averages,
            acquisition_type=AcquisitionType.SPECTROSCOPY,
        ):
            with exp_spec.sweep(uid="resonator_frequency_inner", parameter=inner_sweep):
                for qubit in qubits:
                    # readout pulse and data acquisition
                    with exp_spec.section(uid=f"resonator_spectroscopy_{qubit.uid}"):
                        # resonator signal readout
                        exp_spec.acquire(
                            signal=f"acquire_{qubit.uid}",
                            handle=f"resonator_spectroscopy_{qubit.uid}",
                            length=integration_time,
                        )
                    with exp_spec.section(uid=f"delay_{qubit.uid}", length=1e-6):
                        # holdoff time after signal acquisition
                        exp_spec.reserve(signal=f"measure_{qubit.uid}")

    cal = Calibration()
    local_oscillator = Oscillator(frequency=outer_sweep)
    for qubit in qubits:
        cal[f"measure_{qubit.uid}"] = SignalCalibration(
            oscillator=Oscillator(
                frequency=inner_sweep, modulation_type=ModulationType.HARDWARE
            ),
            local_oscillator=local_oscillator,
            range=measure_range,
        )
        cal[f"acquire_{qubit.uid}"] = SignalCalibration(
            local_oscillator=local_oscillator,
            range=acquire_range,
            port_delay=250e-9,
        )
    exp_spec.set_calibration(cal)

    return exp_spec


def resonator_spectroscopy_single(
    qubit,
    frequency_sweep,
    measure_range=-25,
    acquire_range=-25,
    integration_time=10e-3,
    num_averages=1,
    set_lo=False,
    lo_freq=None,
):
    # Create resonator spectroscopy experiment - uses only readout drive and signal acquisition
    exp_spec = Experiment(
        uid="Resonator Spectroscopy CW Single",
        signals=[
            ExperimentSignal(f"measure_{qubit.uid}", map_to=qubit.signals["measure"]),
            ExperimentSignal(f"acquire_{qubit.uid}", map_to=qubit.signals["acquire"]),
        ],
    )

    ## define experimental sequence
    # loop - average multiple measurements for each frequency - measurement in spectroscopy mode
    with exp_spec.acquire_loop_rt(
        uid="shots",
        count=num_averages,
        acquisition_type=AcquisitionType.SPECTROSCOPY,
    ):
        with exp_spec.sweep(uid="resonator_frequency_inner", parameter=frequency_sweep):
            # readout pulse and data acquisition
            with exp_spec.section(uid=f"resonator_spectroscopy_{qubit.uid}"):
                # resonator signal readout
                exp_spec.acquire(
                    signal=f"acquire_{qubit.uid}",
                    handle=f"resonator_spectroscopy_{qubit.uid}",
                    length=integration_time,
                )
            with exp_spec.section(uid=f"delay_{qubit.uid}", length=1e-6):
                # holdoff time after signal acquisition
                exp_spec.reserve(signal=f"measure_{qubit.uid}")

    cal = Calibration()
    if set_lo and lo_freq is not None:
        local_oscillator = Oscillator(frequency=lo_freq)
    else:
        local_oscillator = Oscillator(frequency=qubit.parameters.readout_lo_frequency)

    cal[f"measure_{qubit.uid}"] = SignalCalibration(
        oscillator=Oscillator(
            frequency=frequency_sweep, modulation_type=ModulationType.HARDWARE
        ),
        local_oscillator=local_oscillator,
        range=measure_range,
    )
    cal[f"acquire_{qubit.uid}"] = SignalCalibration(
        local_oscillator=local_oscillator,
        range=acquire_range,
        port_delay=250e-9,
    )
    exp_spec.set_calibration(cal)

    return exp_spec


def pulsed_resonator_spectroscopy_single(
    qubit,
    frequency_sweep,
    readout_pulse,
    integration_kernel,
    measure_range=-25,
    acquire_range=-25,
    num_averages=1,
    set_lo=False,
    lo_freq=None,
):
    # Create resonator spectroscopy experiment - uses only readout drive and signal acquisition
    exp_spec = Experiment(
        uid="Resonator Spectroscopy Pulsed",
        signals=[
            ExperimentSignal(f"measure_{qubit.uid}", map_to=qubit.signals["measure"]),
            ExperimentSignal(f"acquire_{qubit.uid}", map_to=qubit.signals["acquire"]),
        ],
    )

    ## define experimental sequence
    # loop - average multiple measurements for each frequency - measurement in spectroscopy mode
    with exp_spec.acquire_loop_rt(
        uid="shots",
        count=num_averages,
        acquisition_type=AcquisitionType.SPECTROSCOPY,
    ):
        with exp_spec.sweep(uid="resonator_frequency_inner", parameter=frequency_sweep):
            # readout pulse and data acquisition
            with exp_spec.section(uid=f"resonator_spectroscopy_{qubit.uid}"):
                # resonator signal readout
                exp_spec.measure(
                    measure_signal=f"measure_{qubit.uid}",
                    measure_pulse=readout_pulse(qubit),
                    handle=f"{qubit.uid}_spectroscopy",
                    acquire_signal=f"acquire_{qubit.uid}",
                    integration_kernel=integration_kernel(qubit),
                    reset_delay=qubit.parameters.user_defined["reset_delay_length"],
                )
            with exp_spec.section(uid=f"delay_{qubit.uid}", length=1e-6):
                # holdoff time after signal acquisition
                exp_spec.reserve(signal=f"measure_{qubit.uid}")

    cal = Calibration()
    if set_lo and lo_freq is not None:
        local_oscillator = Oscillator(frequency=lo_freq)
    else:
        local_oscillator = Oscillator(frequency=qubit.parameters.readout_lo_frequency)

    cal[f"measure_{qubit.uid}"] = SignalCalibration(
        oscillator=Oscillator(
            frequency=frequency_sweep, modulation_type=ModulationType.HARDWARE
        ),
        local_oscillator=local_oscillator,
        range=measure_range,
    )
    cal[f"acquire_{qubit.uid}"] = SignalCalibration(
        local_oscillator=local_oscillator,
        range=acquire_range,
        port_delay=250e-9,
    )
    exp_spec.set_calibration(cal)

    return exp_spec


# function that returns a qubit spectroscopy experiment- accepts frequency sweep range as parameter
def qubit_spectroscopy_parallel(
    qubits,
    integration_kernel,
    readout_pulse,
    qubit_spectroscopy_pulse,
    qspec_range=100e6,
    qspec_num=1001,
    num_averages=2**10,
):
    # Create qubit spectroscopy Experiment - uses qubit drive, readout drive and data acquisition lines
    exp_qspec = Experiment(
        uid="Qubit Spectroscopy Parallel",
        signals=flatten(
            [
                [
                    ExperimentSignal(
                        f"drive_{qubit.uid}", map_to=qubit.signals["drive"]
                    ),
                    ExperimentSignal(
                        f"measure_{qubit.uid}", map_to=qubit.signals["measure"]
                    ),
                    ExperimentSignal(
                        f"acquire_{qubit.uid}", map_to=qubit.signals["acquire"]
                    ),
                ]
                for qubit in qubits
            ]
        ),
    )

    # List of frequency sweeps for all qubits
    qubit_frequency_sweeps = [
        LinearSweepParameter(
            uid=f"{qubit.uid}_spectroscopy_sweep",
            start=qubit.parameters.drive_frequency_ge - qspec_range / 2,
            stop=qubit.parameters.drive_frequency_ge + qspec_range / 2,
            count=qspec_num,
            axis_name="Qubit Frequency [Hz]",
        )
        for qubit in qubits
    ]

    # inner loop - real-time averaging - QA in integration mode
    with exp_qspec.acquire_loop_rt(
        uid="freq_shots",
        count=num_averages,
        acquisition_type=AcquisitionType.INTEGRATION,
    ):
        with exp_qspec.sweep(
            uid="qubit_frequency_sweep", parameter=qubit_frequency_sweeps
        ):
            for qubit in qubits:
                # qubit drive
                with exp_qspec.section(uid=f"{qubit.uid}_excitation"):
                    exp_qspec.play(
                        signal=f"drive_{qubit.uid}",
                        pulse=qubit_spectroscopy_pulse(qubit),
                    )
                # measurement
                with exp_qspec.section(
                    uid=f"readout_{qubit.uid}", play_after=f"{qubit.uid}_excitation"
                ):
                    exp_qspec.measure(
                        measure_signal=f"measure_{qubit.uid}",
                        measure_pulse=readout_pulse(qubit),
                        handle=f"{qubit.uid}_spectroscopy",
                        acquire_signal=f"acquire_{qubit.uid}",
                        integration_kernel=integration_kernel(qubit),
                        reset_delay=qubit.parameters.user_defined["reset_delay_length"],
                    )

    cal = Calibration()
    for it, qubit in enumerate(qubits):
        cal[f"drive_{qubit.uid}"] = SignalCalibration(
            oscillator=Oscillator(
                frequency=qubit_frequency_sweeps[it],
                modulation_type=ModulationType.HARDWARE,
            )
        )
    exp_qspec.set_calibration(cal)

    return exp_qspec


# function that returns a qubit spectroscopy experiment- accepts frequency sweep range as parameter
def qubit_spectroscopy_single(
    qubit,
    integration_kernel,
    readout_pulse,
    qubit_spectroscopy_pulse,
    qspec_range=100e6,
    qspec_num=1001,
    num_averages=2**10,
):
    # Create qubit spectroscopy Experiment - uses qubit drive, readout drive and data acquisition lines
    exp_qspec = Experiment(
        uid="Qubit Spectroscopy Single",
        signals=flatten(
            [
                [
                    ExperimentSignal(
                        f"drive_{qubit.uid}", map_to=qubit.signals["drive"]
                    ),
                    ExperimentSignal(
                        f"measure_{qubit.uid}", map_to=qubit.signals["measure"]
                    ),
                    ExperimentSignal(
                        f"acquire_{qubit.uid}", map_to=qubit.signals["acquire"]
                    ),
                ]
            ]
        ),
    )

    # List of frequency sweeps for all qubits
    qubit_frequency_sweep = LinearSweepParameter(
        uid=f"{qubit.uid}_spectroscopy_sweep",
        start=qubit.parameters.drive_frequency_ge - qspec_range / 2,
        stop=qubit.parameters.drive_frequency_ge + qspec_range / 2,
        count=qspec_num,
        axis_name="Qubit Frequency [Hz]",
    )

    # inner loop - real-time averaging - QA in integration mode
    with exp_qspec.acquire_loop_rt(
        uid="freq_shots",
        count=num_averages,
        acquisition_type=AcquisitionType.INTEGRATION,
    ):
        with exp_qspec.sweep(
            uid="qubit_frequency_sweep", parameter=qubit_frequency_sweep
        ):
            # qubit drive
            with exp_qspec.section(uid=f"{qubit.uid}_excitation"):
                exp_qspec.play(
                    signal=f"drive_{qubit.uid}",
                    pulse=qubit_spectroscopy_pulse(qubit),
                )
            # measurement
            with exp_qspec.section(
                uid=f"readout_{qubit.uid}", play_after=f"{qubit.uid}_excitation"
            ):
                exp_qspec.measure(
                    measure_signal=f"measure_{qubit.uid}",
                    measure_pulse=readout_pulse(qubit),
                    handle=f"{qubit.uid}_spectroscopy",
                    acquire_signal=f"acquire_{qubit.uid}",
                    integration_kernel=integration_kernel(qubit),
                    reset_delay=qubit.parameters.user_defined["reset_delay_length"],
                )

    cal = Calibration()
    cal[f"drive_{qubit.uid}"] = SignalCalibration(
        oscillator=Oscillator(
            frequency=qubit_frequency_sweep,
            modulation_type=ModulationType.HARDWARE,
        )
    )
    exp_qspec.set_calibration(cal)

    return exp_qspec


# function that defines a amplitude vs frequency resonator spectroscopy experiment
def res_spectroscopy_pulsed_amp_sweep(
    qubit,
    integration_kernel,
    readout_pulse,
    frequency_sweep,
    amplitude_sweep,
    num_averages,
    measure_range,
    acquire_range,
):
    # Create resonator spectroscopy experiment - uses only readout drive and signal acquisition
    exp_spec = Experiment(
        uid="Resonator Spectroscopy with Amp",
        signals=flatten(
            [
                [
                    ExperimentSignal(
                        f"measure_{qubit.uid}", map_to=qubit.signals["measure"]
                    ),
                    ExperimentSignal(
                        f"acquire_{qubit.uid}", map_to=qubit.signals["acquire"]
                    ),
                ]
            ]
        ),
    )

    ## define experimental sequence
    # outer loop - vary drive frequency
    with exp_spec.sweep(uid="res_amp", parameter=amplitude_sweep):
        with exp_spec.acquire_loop_rt(
            uid="shots",
            count=num_averages,
            acquisition_type=AcquisitionType.SPECTROSCOPY,
        ):
            with exp_spec.sweep(
                uid="resonator_frequency_inner", parameter=frequency_sweep
            ):
                # readout pulse and data acquisition
                with exp_spec.section(uid=f"resonator_spectroscopy_{qubit.uid}"):
                    # resonator signal readout
                    exp_spec.measure(
                        measure_signal=f"measure_{qubit.uid}",
                        measure_pulse=readout_pulse(qubit),
                        handle=f"{qubit.uid}_spectroscopy",
                        acquire_signal=f"acquire_{qubit.uid}",
                        integration_kernel=integration_kernel(qubit),
                        reset_delay=qubit.parameters.user_defined["reset_delay_length"],
                    )
                #     # play resonator excitation pulse
                #     exp_spec.play(
                #         signal=f"measure_{qubit.uid}", pulse=readout_pulse(qubit)
                #     )
                #     # resonator signal readout
                #     exp_spec.acquire(
                #         signal=f"acquire_{qubit.uid}",
                #         handle=f"{qubit.uid}_spectroscopy",
                #         length=qubit.parameters.user_defined["readout_length"],
                #     )
                # with exp_spec.section(uid=f"delay_{qubit.uid}", length=1e-6):
                #     # holdoff time after signal acquisition
                #     exp_spec.reserve(signal=f"measure_{qubit.uid}")

    cal = Calibration()
    cal[f"measure_{qubit.uid}"] = SignalCalibration(
        oscillator=Oscillator(
            frequency=frequency_sweep,
            modulation_type=ModulationType.HARDWARE,
        ),
        amplitude=amplitude_sweep,
        range=measure_range,
    )
    cal[f"acquire_{qubit.uid}"] = SignalCalibration(
        range=acquire_range,
        port_delay=250e-9,
    )
    exp_spec.set_calibration(cal)

    return exp_spec


def resonator_spectroscopy_single_CW_vs_power(
    qubit,
    amp_sweep,
    inner_sweep,
    acquire_range=5,
    measure_range=-25,
    integration_time=10e-3,
    num_averages=2**10,
):
    # Create resonator spectroscopy experiment - uses only readout drive and signal acquisition
    exp_spec = Experiment(
        uid="Resonator Spectroscopy",
        signals=[
            signal
            for signal_list in [
                [
                    ExperimentSignal(
                        f"measure_{qubit.uid}", map_to=qubit.signals["measure"]
                    ),
                    ExperimentSignal(
                        f"acquire_{qubit.uid}", map_to=qubit.signals["acquire"]
                    ),
                ]
                # for qubit in qubits
            ]
            for signal in signal_list
        ],
    )

    ## define experimental sequence
    # loop - average multiple measurements for each frequency - measurement in spectroscopy mode
    with exp_spec.sweep(uid="resonator_frequency_outer", parameter=amp_sweep):
        with exp_spec.acquire_loop_rt(
            uid="shots",
            count=num_averages,
            acquisition_type=AcquisitionType.SPECTROSCOPY,
        ):
            with exp_spec.sweep(uid="resonator_frequency_inner", parameter=inner_sweep):
                # for qubit in qubits:
                # readout pulse and data acquisition
                with exp_spec.section(uid=f"resonator_spectroscopy_{qubit.uid}"):
                    # resonator signal readout
                    exp_spec.acquire(
                        signal=f"acquire_{qubit.uid}",
                        handle=f"resonator_spectroscopy_{qubit.uid}",
                        length=integration_time,
                    )
                with exp_spec.section(uid=f"delay_{qubit.uid}", length=1e-6):
                    # holdoff time after signal acquisition
                    exp_spec.reserve(signal=f"measure_{qubit.uid}")

    cal = Calibration()
    # for qubit in qubits:
    cal[f"measure_{qubit.uid}"] = SignalCalibration(
        oscillator=Oscillator(
            frequency=inner_sweep, modulation_type=ModulationType.HARDWARE
        ),
        amplitude=amp_sweep,
        range=measure_range,
    )
    cal[f"acquire_{qubit.uid}"] = SignalCalibration(
        range=acquire_range,
        port_delay=250e-9,
    )
    exp_spec.set_calibration(cal)

    return exp_spec


# function that returns an amplitude Rabi experiment
def amplitude_rabi_parallel(
    qubits,
    drive_pulse,
    integration_kernel,
    readout_pulse,
    amplitude_sweep,
    num_averages=2**10,
):
    exp_rabi = Experiment(
        uid="Rabi Parallel",
        signals=flatten(
            [
                [
                    ExperimentSignal(
                        f"drive_{qubit.uid}", map_to=qubit.signals["drive"]
                    ),
                    ExperimentSignal(
                        f"measure_{qubit.uid}", map_to=qubit.signals["measure"]
                    ),
                    ExperimentSignal(
                        f"acquire_{qubit.uid}", map_to=qubit.signals["acquire"]
                    ),
                ]
                for qubit in qubits
            ]
        ),
    )

    ## define Rabi experiment pulse sequence
    # outer loop - real-time, cyclic averaging
    with exp_rabi.acquire_loop_rt(
        uid="rabi_shots",
        count=num_averages,
        averaging_mode=AveragingMode.CYCLIC,
        acquisition_type=AcquisitionType.INTEGRATION,
    ):
        # inner loop - real time sweep of Rabi ampitudes
        with exp_rabi.sweep(uid="rabi_sweep", parameter=amplitude_sweep):
            for qubit in qubits:
                # qubit drive
                with exp_rabi.section(
                    uid=f"{qubit.uid}_excitation", alignment=SectionAlignment.RIGHT
                ):
                    exp_rabi.play(
                        signal=f"drive_{qubit.uid}",
                        pulse=drive_pulse(qubit),
                        amplitude=amplitude_sweep,
                    )
                # measurement
                with exp_rabi.section(
                    uid=f"readout_{qubit.uid}", play_after=f"{qubit.uid}_excitation"
                ):
                    exp_rabi.measure(
                        measure_signal=f"measure_{qubit.uid}",
                        measure_pulse=readout_pulse(qubit),
                        handle=f"{qubit.uid}_rabi",
                        acquire_signal=f"acquire_{qubit.uid}",
                        integration_kernel=integration_kernel(qubit),
                        reset_delay=qubit.parameters.user_defined["reset_delay_length"],
                    )

    return exp_rabi


def amplitude_rabi_single(
    qubit,
    drive_pulse,
    integration_kernel,
    readout_pulse,
    amplitude_sweep,
    num_averages=2**10,
    cal_trace=False,
    pi_amplitude=0.5,
):
    exp_rabi = Experiment(
        uid="Qubit Spectroscopy",
        signals=flatten(
            [
                [
                    ExperimentSignal(
                        f"drive_{qubit.uid}", map_to=qubit.signals["drive"]
                    ),
                    ExperimentSignal(
                        f"measure_{qubit.uid}", map_to=qubit.signals["measure"]
                    ),
                    ExperimentSignal(
                        f"acquire_{qubit.uid}", map_to=qubit.signals["acquire"]
                    ),
                ]
            ]
        ),
    )

    ## define Rabi experiment pulse sequence
    # outer loop - real-time, cyclic averaging
    with exp_rabi.acquire_loop_rt(
        uid="rabi_shots",
        count=num_averages,
        averaging_mode=AveragingMode.CYCLIC,
        acquisition_type=AcquisitionType.INTEGRATION,
    ):
        # inner loop - real time sweep of Rabi amplitudes
        with exp_rabi.sweep(uid="rabi_sweep", parameter=amplitude_sweep):
            # qubit drive
            with exp_rabi.section(
                uid=f"{qubit.uid}_excitation", alignment=SectionAlignment.RIGHT
            ):
                exp_rabi.play(
                    signal=f"drive_{qubit.uid}",
                    pulse=drive_pulse(qubit),
                    amplitude=amplitude_sweep,
                )
            # measurement
            with exp_rabi.section(
                uid=f"readout_{qubit.uid}", play_after=f"{qubit.uid}_excitation"
            ):
                exp_rabi.measure(
                    measure_signal=f"measure_{qubit.uid}",
                    measure_pulse=readout_pulse(qubit),
                    handle=f"{qubit.uid}_rabi",
                    acquire_signal=f"acquire_{qubit.uid}",
                    integration_kernel=integration_kernel(qubit),
                    reset_delay=qubit.parameters.user_defined["reset_delay_length"],
                )
        if cal_trace:
            with exp_rabi.section(uid="cal_trace_gnd"):
                exp_rabi.measure(
                    measure_signal=f"measure_{qubit.uid}",
                    measure_pulse=readout_pulse(qubit),
                    handle=f"{qubit.uid}_rabi_cal_trace",
                    acquire_signal=f"acquire_{qubit.uid}",
                    integration_kernel=integration_kernel(qubit),
                    reset_delay=qubit.parameters.user_defined["reset_delay_length"],
                )
            with exp_rabi.section(uid="cal_trace_exc"):
                exp_rabi.play(
                    signal=f"drive_{qubit.uid}",
                    pulse=drive_pulse(qubit),
                    amplitude=pi_amplitude,
                )

                exp_rabi.measure(
                    measure_signal=f"measure_{qubit.uid}",
                    measure_pulse=readout_pulse(qubit),
                    handle=f"{qubit.uid}_rabi_cal_trace",
                    acquire_signal=f"acquire_{qubit.uid}",
                    integration_kernel=integration_kernel(qubit),
                    reset_delay=qubit.parameters.user_defined["reset_delay_length"],
                )

    return exp_rabi


# function that returns a ramsey experiment
def ramsey_parallel(
    qubits,
    drive_pulse: callable,
    integration_kernel: callable,
    readout_pulse: callable,
    delay_sweep,
    num_averages=2**10,
    cal_trace=False,
    pi_amplitude=0.5,
):
    exp_ramsey = Experiment(
        uid="Ramsey Exp",
        signals=flatten(
            [
                [
                    ExperimentSignal(
                        f"drive_{qubit.uid}", map_to=qubit.signals["drive"]
                    ),
                    ExperimentSignal(
                        f"measure_{qubit.uid}", map_to=qubit.signals["measure"]
                    ),
                    ExperimentSignal(
                        f"acquire_{qubit.uid}", map_to=qubit.signals["acquire"]
                    ),
                ]
                for qubit in qubits
            ]
        ),
    )

    ## define Ramsey experiment pulse sequence
    # outer loop - real-time, cyclic averaging
    with exp_ramsey.acquire_loop_rt(
        uid="ramsey_shots",
        count=num_averages,
        averaging_mode=AveragingMode.CYCLIC,
        acquisition_type=AcquisitionType.INTEGRATION,
        repetition_mode=RepetitionMode.AUTO,
    ):
        # inner loop - real time sweep of Ramsey time delays
        with exp_ramsey.sweep(
            uid="ramsey_sweep", parameter=delay_sweep, alignment=SectionAlignment.RIGHT
        ):
            for qubit in qubits:
                # play qubit excitation pulse - pulse amplitude is swept
                ramsey_pulse = drive_pulse(qubit)
                with exp_ramsey.section(
                    uid=f"{qubit.uid}_excitation", alignment=SectionAlignment.RIGHT
                ):
                    exp_ramsey.play(signal=f"drive_{qubit.uid}", pulse=ramsey_pulse)
                    exp_ramsey.delay(signal=f"drive_{qubit.uid}", time=delay_sweep)
                    exp_ramsey.play(signal=f"drive_{qubit.uid}", pulse=ramsey_pulse)
                # readout pulse and data acquisition
                # measurement
                with exp_ramsey.section(
                    uid=f"readout_{qubit.uid}", play_after=f"{qubit.uid}_excitation"
                ):
                    exp_ramsey.measure(
                        measure_signal=f"measure_{qubit.uid}",
                        measure_pulse=readout_pulse(qubit),
                        handle=f"{qubit.uid}_ramsey",
                        acquire_signal=f"acquire_{qubit.uid}",
                        integration_kernel=integration_kernel(qubit),
                        reset_delay=qubit.parameters.user_defined["reset_delay_length"],
                    )

                if cal_trace:
                    with exp_ramsey.section(uid="cal_trace_gnd"):
                        exp_ramsey.measure(
                            measure_signal=f"measure_{qubit.uid}",
                            measure_pulse=readout_pulse(qubit),
                            handle=f"{qubit.uid}_rabi_cal_trace",
                            acquire_signal=f"acquire_{qubit.uid}",
                            integration_kernel=integration_kernel(qubit),
                            reset_delay=qubit.parameters.user_defined[
                                "reset_delay_length"
                            ],
                        )
                    with exp_ramsey.section(uid="cal_trace_exc"):
                        exp_ramsey.play(
                            signal=f"drive_{qubit.uid}",
                            pulse=drive_pulse(qubit),
                            amplitude=pi_amplitude,
                        )

                        exp_ramsey.measure(
                            measure_signal=f"measure_{qubit.uid}",
                            measure_pulse=readout_pulse(qubit),
                            handle=f"{qubit.uid}_rabi_cal_trace",
                            acquire_signal=f"acquire_{qubit.uid}",
                            integration_kernel=integration_kernel(qubit),
                            reset_delay=qubit.parameters.user_defined[
                                "reset_delay_length"
                            ],
                        )

    return exp_ramsey


# function that returns a t1 experiment
def t1_parallel(
    qubits,
    drive_pulse,
    integration_kernel,
    readout_pulse,
    delay_sweep,
    num_averages=2**10,
):
    exp_t1 = Experiment(
        uid="T1 Exp",
        signals=flatten(
            [
                [
                    ExperimentSignal(
                        f"drive_{qubit.uid}", map_to=qubit.signals["drive"]
                    ),
                    ExperimentSignal(
                        f"measure_{qubit.uid}", map_to=qubit.signals["measure"]
                    ),
                    ExperimentSignal(
                        f"acquire_{qubit.uid}", map_to=qubit.signals["acquire"]
                    ),
                ]
                for qubit in qubits
            ]
        ),
    )

    ## define Ramsey experiment pulse sequence
    # outer loop - real-time, cyclic averaging
    with exp_t1.acquire_loop_rt(
        uid="t1_shots",
        count=num_averages,
        averaging_mode=AveragingMode.CYCLIC,
        acquisition_type=AcquisitionType.INTEGRATION,
        repetition_mode=RepetitionMode.AUTO,
    ):
        # inner loop - real time sweep of T1 time delays
        with exp_t1.sweep(
            uid="t1_delay_sweep",
            parameter=delay_sweep,
            alignment=SectionAlignment.RIGHT,
        ):
            for qubit in qubits:
                # play qubit excitation pulse - pulse amplitude is swept
                with exp_t1.section(
                    uid=f"{qubit.uid}_excitation", alignment=SectionAlignment.RIGHT
                ):
                    exp_t1.play(signal=f"drive_{qubit.uid}", pulse=drive_pulse(qubit))
                    exp_t1.delay(signal=f"drive_{qubit.uid}", time=delay_sweep)
                # readout pulse and data acquisition
                # measurement
                with exp_t1.section(
                    uid=f"readout_{qubit.uid}", play_after=f"{qubit.uid}_excitation"
                ):
                    exp_t1.measure(
                        measure_signal=f"measure_{qubit.uid}",
                        measure_pulse=readout_pulse(qubit),
                        handle=f"{qubit.uid}_t1",
                        acquire_signal=f"acquire_{qubit.uid}",
                        integration_kernel=integration_kernel(qubit),
                        reset_delay=qubit.parameters.user_defined["reset_delay_length"],
                    )

    return exp_t1


# function that returns a ecr amplitude sweep experiment
# This experiment currently relies on the drive_ef lines
# to be easily compatible with the transmon objects.
# TODO: Update with CR lines
def ecr_amplitude_sweep(
    control_qubit,
    target_qubit,
    drive_pulse,
    cancellation_length,
    control_cr_freq,
    target_cr_freq,
    target_amplitude,
    integration_kernel,
    readout_pulse,
    amplitude_sweep,
    num_averages=2**10,
):
    exp_ecr_amp = Experiment(
        uid="ecr_amp",
        signals=flatten(
            [
                [
                    ExperimentSignal(
                        f"drive_{control_qubit.uid}",
                        map_to=control_qubit.signals["drive"],
                    ),
                    ExperimentSignal(
                        f"drive_ef_{control_qubit.uid}",
                        map_to=control_qubit.signals["drive_ef"],
                    ),
                    ExperimentSignal(
                        f"measure_{control_qubit.uid}",
                        map_to=control_qubit.signals["measure"],
                    ),
                    ExperimentSignal(
                        f"acquire_{control_qubit.uid}",
                        map_to=control_qubit.signals["acquire"],
                    ),
                    ExperimentSignal(
                        f"drive_{target_qubit.uid}",
                        map_to=target_qubit.signals["drive"],
                    ),
                    ExperimentSignal(
                        f"drive_ef_{target_qubit.uid}",
                        map_to=target_qubit.signals["drive_ef"],
                    ),
                    ExperimentSignal(
                        f"measure_{target_qubit.uid}",
                        map_to=target_qubit.signals["measure"],
                    ),
                    ExperimentSignal(
                        f"acquire_{target_qubit.uid}",
                        map_to=target_qubit.signals["acquire"],
                    ),
                ]
            ]
        ),
    )
    ## define Rabi experiment pulse sequence
    # outer loop - real-time, cyclic averaging
    with exp_ecr_amp.acquire_loop_rt(
        uid="ecr_shots",
        count=num_averages,
        averaging_mode=AveragingMode.CYCLIC,
        acquisition_type=AcquisitionType.INTEGRATION,
        reset_oscillator_phase=True,
    ):
        # inner loop - real time sweep of Rabi ampitudes
        with exp_ecr_amp.sweep(uid="rabi_sweep", parameter=amplitude_sweep):
            with exp_ecr_amp.section(uid="excitation"):
                # create ecr gate
                ecr_id = f"ecr_{control_qubit.uid}_{target_qubit.uid}"

                gate = Section(
                    uid=id_generator(ecr_id),
                    alignment=SectionAlignment.RIGHT,
                    on_system_grid=True,
                )

                # define X pulses for target and control
                x180_pulse_control = drive_pulse(control_qubit)

                # define cancellation pulses for target and control
                cancellation_control_n = pulse_library.gaussian_square(
                    uid="CR-",
                    length=cancellation_length,
                    width=0.95 * cancellation_length,
                    phase=np.pi,
                )
                cancellation_control_p = pulse_library.gaussian_square(
                    uid="CR+",
                    length=cancellation_length,
                    width=0.95 * cancellation_length,
                )
                cancellation_target_p = pulse_library.gaussian_square(
                    uid="q1+",
                    length=cancellation_length,
                    width=0.95 * cancellation_length,
                )
                cancellation_target_n = pulse_library.gaussian_square(
                    uid="q1-",
                    length=cancellation_length,
                    width=0.95 * cancellation_length,
                    phase=np.pi,
                )

                # First cross-resonance component
                cancellation_p = Section(
                    uid=id_generator(f"{ecr_id}_canc_p"),
                    on_system_grid=True,
                )
                cancellation_p.play(
                    signal=f"drive_ef_{target_qubit.uid}",
                    pulse=cancellation_target_p,
                    amplitude=target_amplitude,
                )
                cancellation_p.play(
                    signal=f"drive_ef_{control_qubit.uid}",
                    pulse=cancellation_control_p,
                    amplitude=amplitude_sweep,
                )
                gate.add(cancellation_p)

                # play X pulse on control
                x180_control = Section(
                    uid=id_generator(f"{ecr_id}_x_180_control"),
                    play_after=cancellation_p.uid,
                    on_system_grid=True,
                )
                x180_control.play(
                    signal=f"drive_{control_qubit.uid}", pulse=x180_pulse_control
                )
                gate.add(x180_control)

                # Second cross-resonance component
                cancellation_n = Section(
                    uid=id_generator(f"ecr_{ecr_id}_canc_n"),
                    play_after=x180_control.uid,
                    on_system_grid=True,
                )
                cancellation_n.play(
                    signal=f"drive_ef_{target_qubit.uid}",
                    pulse=cancellation_target_n,
                    amplitude=target_amplitude,
                    phase=np.pi,
                    # increment_oscillator_phase=np.pi,
                )
                cancellation_n.play(
                    signal=f"drive_ef_{control_qubit.uid}",
                    pulse=cancellation_control_n,
                    amplitude=amplitude_sweep,
                    phase=np.pi,
                    # increment_oscillator_phase=np.pi,
                )
                cancellation_n.play(
                    signal=f"drive_ef_{target_qubit.uid}",
                    pulse=None,
                    # increment_oscillator_phase=np.pi,
                )
                gate.add(cancellation_n)

                # play second X pulse on control
                x180_control_2 = Section(
                    uid=id_generator(f"{ecr_id}_x_180_control_2"),
                    play_after=cancellation_n.uid,
                    on_system_grid=True,
                )
                x180_control_2.play(
                    signal=f"drive_{control_qubit.uid}",
                    pulse=x180_pulse_control,
                    # increment_oscillator_phase=np.pi,
                )
                gate.add(x180_control_2)

                # # add the section
                exp_ecr_amp.add(gate)
            # measurement
            for qubit in [control_qubit, target_qubit]:
                with exp_ecr_amp.section(
                    uid=f"readout_{qubit.uid}",
                    play_after="excitation",
                    on_system_grid=True,
                ):
                    exp_ecr_amp.measure(
                        measure_signal=f"measure_{qubit.uid}",
                        measure_pulse=readout_pulse(qubit),
                        handle=f"{qubit.uid}_ecr_amp",
                        acquire_signal=f"acquire_{qubit.uid}",
                        integration_kernel=integration_kernel(qubit),
                        reset_delay=qubit.parameters.user_defined["reset_delay_length"],
                    )
            for qubit in [control_qubit, target_qubit]:
                with exp_ecr_amp.section(
                    uid=f"buffer_{qubit.uid}",
                    length=4e-9,
                    on_system_grid=True,
                ):
                    exp_ecr_amp.reserve(signal=f"measure_{qubit.uid}")
                    exp_ecr_amp.reserve(signal=f"drive_{qubit.uid}")

    cal = Calibration()
    cal[f"drive_ef_{control_qubit.uid}"] = SignalCalibration(
        oscillator=Oscillator(
            frequency=control_cr_freq,
            modulation_type=ModulationType.HARDWARE,
        ),
    )
    cal[f"drive_ef_{target_qubit.uid}"] = SignalCalibration(
        oscillator=Oscillator(
            frequency=target_cr_freq,
            modulation_type=ModulationType.HARDWARE,
        ),
    )
    exp_ecr_amp.set_calibration(cal)

    return exp_ecr_amp


def plot_with_trace(res):
    handles = list(res.acquired_results.keys())
    res1 = np.asarray(res.get_data(handles[0]))
    res2 = np.asarray(res.get_data(handles[1]))
    axis1 = res.get_axis(handles[0])[0]
    delta_x = axis1[-1] - axis1[-2]
    axis2 = np.linspace(axis1[-1] + delta_x, axis1[-1] + 2 * delta_x, 2)
    axis = np.concatenate((axis1, axis2)).flatten()
    temp_res = np.concatenate((res1, res2)).flatten()
    # plt.plot(axis,np.abs(temp_res),'-o')

    delta_vec = res2[0] - res2[1]
    angle = np.angle(delta_vec)
    rd = []
    for r in [res1, res2]:
        r = r - res2[0]
        r = np.asarray(r) * np.exp(-1j * angle)
        r = r / np.abs(delta_vec)
        rd.append(r)

    temp_rd = np.concatenate((rd[0], rd[1])).flatten()
    plt.plot(axis, np.abs(temp_rd), "-o")


def resonator_spectroscopy_g_vs_e(
    qubit,
    drive_pulse,
    readout_pulse,
    integration_kernel,
    frequency_sweep,
    measure_range=-25,
    acquire_range=-25,
    num_averages=2**8,
    set_lo=False,
    lo_freq=None,
):
    # Create resonator spectroscopy experiment - uses only readout drive and signal acquisition
    exp_spec = Experiment(
        uid="Resonator Spectroscopy CW Single",
        signals=[
            ExperimentSignal(f"measure_{qubit.uid}", map_to=qubit.signals["measure"]),
            ExperimentSignal(f"acquire_{qubit.uid}", map_to=qubit.signals["acquire"]),
            ExperimentSignal(f"drive_{qubit.uid}", map_to=qubit.signals["drive"]),
        ],
    )

    ## define experimental sequence
    # loop - average multiple measurements for each frequency - measurement in spectroscopy mode
    with exp_spec.acquire_loop_rt(
        uid="shots",
        count=num_averages,
        acquisition_type=AcquisitionType.SPECTROSCOPY,
    ):
        if drive_pulse is None:
            with exp_spec.sweep(uid="resonator_frequency_g", parameter=frequency_sweep):
                # readout pulse and data acquisition

                with exp_spec.section(uid=f"resonator_spectroscopy_{qubit.uid}_g"):
                    # resonator signal readout
                    exp_spec.measure(
                        measure_signal=f"measure_{qubit.uid}",
                        measure_pulse=readout_pulse(qubit),
                        handle=f"resonator_spectroscopy_{qubit.uid}",
                        acquire_signal=f"acquire_{qubit.uid}",
                        integration_kernel=integration_kernel(qubit),
                        reset_delay=1e-6,
                    )
                with exp_spec.section(uid=f"delay_{qubit.uid}_g", length=1e-6):
                    exp_spec.reserve(signal=f"measure_{qubit.uid}")
                    exp_spec.reserve(signal=f"acquire_{qubit.uid}")

        # excited
        else:
            with exp_spec.sweep(uid="resonator_frequency_e", parameter=frequency_sweep):
                with exp_spec.section(uid="excitation"):
                    exp_spec.play(signal=f"drive_{qubit.uid}", pulse=drive_pulse(qubit))
                with exp_spec.section(
                    uid=f"resonator_spectroscopy_{qubit.uid}_e", play_after="excitation"
                ):
                    # resonator signal readout
                    exp_spec.measure(
                        measure_signal=f"measure_{qubit.uid}",
                        measure_pulse=readout_pulse(qubit),
                        handle=f"resonator_spectroscopy_{qubit.uid}",
                        acquire_signal=f"acquire_{qubit.uid}",
                        integration_kernel=integration_kernel(qubit),
                        reset_delay=qubit.parameters.user_defined["reset_delay_length"],
                    )
                with exp_spec.section(uid=f"delay_{qubit.uid}_e", length=1e-6):
                    exp_spec.reserve(signal=f"measure_{qubit.uid}")
                    exp_spec.reserve(signal=f"acquire_{qubit.uid}")

    cal = Calibration()
    if set_lo and lo_freq is not None:
        local_oscillator = Oscillator(frequency=lo_freq)
    else:
        local_oscillator = Oscillator(frequency=qubit.parameters.readout_lo_frequency)

    cal[f"measure_{qubit.uid}"] = SignalCalibration(
        oscillator=Oscillator(
            frequency=frequency_sweep, modulation_type=ModulationType.HARDWARE
        ),
        local_oscillator=local_oscillator,
        range=measure_range,
    )
    cal[f"acquire_{qubit.uid}"] = SignalCalibration(
        local_oscillator=local_oscillator,
        range=acquire_range,
        port_delay=250e-9,
    )
    exp_spec.set_calibration(cal)

    return exp_spec
