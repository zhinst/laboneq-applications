from laboneq.dsl.experiment.builtins import *  # noqa: F403
from laboneq.simple import *  # noqa: F403

from . import quantum_operations as qt_ops
from laboneq_library.calibration_helpers import \
    update_setup_calibration_from_qubits
import time
import json
import os


@experiment(signals=["measure", "acquire"], uid="Full range CW resonator sweep")
def resonator_spectroscopy_full_range(
    qubit,
    lo_frequencies,
    inner_start,
    inner_stop,
    inner_count,
    cw=True,
    integration_time=10e-3,
    num_averages=1,
):
    map_signal("measure", qubit.signals["measure"])
    map_signal("acquire", qubit.signals["acquire"])

    cal = experiment_calibration()
    local_oscillator = Oscillator()
    inner_oscillator = Oscillator(modulation_type=ModulationType.HARDWARE)
    cal["measure"] = SignalCalibration(
        oscillator=inner_oscillator,
        local_oscillator=local_oscillator,
        range=qubit.parameters.readout_range_out,
    )
    cal["acquire"] = SignalCalibration(
        local_oscillator=local_oscillator,
        oscillator=inner_oscillator,
        range=qubit.parameters.readout_range_in,
        port_delay=250e-9,
    )

    with for_each(lo_frequencies, axis_name="outer_sweep") as lo_freq:
        local_oscillator.frequency = lo_freq

        with acquire_loop_rt(
            uid="shots",
            count=num_averages,
            acquisition_type=AcquisitionType.SPECTROSCOPY,
        ):
            with sweep_range(
                start=inner_start,
                stop=inner_stop,
                count=inner_count,
                axis_name="inner_sweep",
            ) as inner_freq:
                inner_oscillator.frequency = inner_freq

                # readout pulse and data acquisition
                with section(uid="resonator_spectroscopy"):
                    if cw:
                        acquire(
                            signal="acquire",
                            handle="resonator_spectroscopy",
                            length=integration_time,
                        )
                    else:
                        kernel_length = max(2e-6, integration_time)
                        play(
                            signal="measure",
                            pulse=pulse_library.const(length=kernel_length),
                        )
                        acquire(
                            signal="acquire",
                            handle="resonator_spectroscopy",
                            length=kernel_length,
                        )
                with section(
                    uid="delay", length=1e-6, play_after="resonator_spectroscopy"
                ):
                    pass


@experiment(signals=["drive", "measure", "acquire"], uid="Amplitude Rabi Experiment")
def amplitude_rabi_single(
    qubit,
    amplitude_sweep,
    num_averages=2**10,
    cal_trace=False,
):
    map_signal("drive", qubit.signals["drive"])
    map_signal("measure", qubit.signals["measure"])
    map_signal("acquire", qubit.signals["acquire"])

    ## define Rabi experiment pulse sequence
    # outer loop - real-time, cyclic averaging
    with acquire_loop_rt(
        uid="rabi_shots",
        count=num_averages,
        averaging_mode=AveragingMode.CYCLIC,
        acquisition_type=AcquisitionType.INTEGRATION,
    ):
        # inner loop - real time sweep of Rabi amplitudes
        with sweep(uid="rabi_sweep", parameter=amplitude_sweep):
            # qubit drive
            with section(uid="excitation", alignment=SectionAlignment.RIGHT):
                play(
                    signal="drive",
                    pulse=qt_ops.drive_ge_pi(qubit, amplitude=1),
                    amplitude=amplitude_sweep,
                )
                """
                    Unfortunately, we are not yet ready to do something like
                    add(quantum_operations.drive_ge(qubit)).
                    We need to design a best way to come up with a way how to propagate
                    parameters through, if they are convoluted as in the amp rabi sweep.
                """

            # measurement
            with section(uid="readout", play_after="excitation"):
                readout_pulse = qt_ops.readout_pulse(qubit)
                integration_kernel = qt_ops.integration_kernel(qubit)

                measure(
                    measure_signal="measure",
                    measure_pulse=readout_pulse,
                    handle="rabi",
                    acquire_signal="acquire",
                    integration_kernel=integration_kernel,
                    reset_delay=qubit.parameters.user_defined["reset_delay_length"],
                )
        if cal_trace:
            with section(uid="cal_trace_gnd_meas"):
                measure(
                    measure_signal="measure",
                    measure_pulse=readout_pulse,
                    handle=f"{qubit.uid}_rabi_cal_trace",
                    acquire_signal="acquire",
                    integration_kernel=integration_kernel,
                    reset_delay=1e-6,
                )
            with section(uid="cal_trace_exc", play_after="cal_trace_gnd_meas"):
                play(signal="drive", pulse=qt_ops.drive_ge_pi(qubit))

            with section(uid="cal_trace_exc_meas", play_after="cal_trace_exc"):
                measure(
                    measure_signal="measure",
                    measure_pulse=readout_pulse,
                    handle="rabi_cal_trace",
                    acquire_signal="acquire",
                    integration_kernel=integration_kernel,
                    reset_delay=qubit.parameters.user_defined["reset_delay_length"],
                )


def ramsey_parallel(
    qubits,
    delay_sweep,
    num_averages=2**10,
    detuning=0,
    cal_trace=False,
):

    signal_list = []
    signal_types = ["drive", "measure", "acquire"]

    def signal_name(signal, qubit):
        return f"{signal}_{qubit.uid}"

    for qubit in qubits:
        for signal in signal_types:
            signal_list.append(signal_name(signal, qubit))

    @experiment(signals=signal_list, uid="ramsey_parallel")
    def exp_ramsey():
        # map all lines
        for qubit in qubits:
            for signal in signal_types:
                map_signal(signal_name(signal, qubit), qubit.signals[signal])

        # include detuning
        freqs = [
            qubit.parameters.resonance_frequency_ge
            + detuning
            - qubit.parameters.drive_lo_frequency
            for qubit in qubits
        ]

        calibration = experiment_calibration()

        for i, qubit in enumerate(qubits):
            calibration[signal_name("drive", qubit)] = SignalCalibration(
                oscillator=Oscillator(
                    frequency=freqs[i], modulation_type=ModulationType.HARDWARE
                )
            )

        ## define Ramsey experiment pulse sequence
        # outer loop - real-time, cyclic averaging
        with acquire_loop_rt(
            uid="ramsey_shots",
            count=num_averages,
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION,
        ):
            for qubit in qubits:
                # inner loop - real time sweep of Ramsey time delays
                with sweep(
                    uid=f"ramsey_sweep_{qubit.uid}",
                    parameter=delay_sweep,
                    alignment=SectionAlignment.RIGHT,
                ):
                    with section(
                        uid=f"{qubit.uid}_excitation", alignment=SectionAlignment.RIGHT
                    ):
                        ramsey_drive_pulse = qt_ops.drive_ge_pi2(qubit)
                        play(signal=f"drive_{qubit.uid}", pulse=ramsey_drive_pulse)
                        delay(signal=f"drive_{qubit.uid}", time=delay_sweep)
                        play(signal=f"drive_{qubit.uid}", pulse=ramsey_drive_pulse)

                    # readout pulse and data acquisition
                    # measurement
                    with section(
                        uid=f"readout_{qubit.uid}", play_after=f"{qubit.uid}_excitation"
                    ):
                        measure(
                            measure_signal=f"measure_{qubit.uid}",
                            measure_pulse=qt_ops.readout_pulse(qubit),
                            handle=f"{qubit.uid}_ramsey",
                            acquire_signal=f"acquire_{qubit.uid}",
                            integration_kernel=qt_ops.integration_kernel(
                                qubit
                            ),
                            reset_delay=qubit.parameters.user_defined[
                                "reset_delay_length"
                            ],
                        )

                if cal_trace:
                    with section(
                        uid=f"cal_trace_gnd_{qubit.uid}",
                        play_after=f"ramsey_sweep_{qubit.uid}",
                    ):
                        measure(
                            measure_signal=f"measure_{qubit.uid}",
                            measure_pulse=qt_ops.readout_pulse(qubit),
                            handle=f"{qubit.uid}_ramsey_cal_trace",
                            acquire_signal=f"acquire_{qubit.uid}",
                            integration_kernel=qt_ops.integration_kernel(
                                qubit
                            ),
                            reset_delay=1e-6,  # qubit.parameters.user_defined["reset_delay_length"],
                        )

                    with section(
                        uid=f"cal_trace_exc_{qubit.uid}",
                        play_after=f"cal_trace_gnd_{qubit.uid}",
                    ):
                        play(
                            signal=f"drive_{qubit.uid}",
                            pulse=qt_ops.drive_ge_pi(qubit),
                        )

                    with section(
                        uid=f"cal_trace_exc_meas_{qubit.uid}",
                        play_after=f"cal_trace_exc_{qubit.uid}",
                    ):
                        measure(
                            measure_signal=f"measure_{qubit.uid}",
                            measure_pulse=qt_ops.readout_pulse(qubit),
                            handle=f"{qubit.uid}_ramsey_cal_trace",
                            acquire_signal=f"acquire_{qubit.uid}",
                            integration_kernel=qt_ops.integration_kernel(
                                qubit
                            ),
                            reset_delay=qubit.parameters.user_defined[
                                "reset_delay_length"
                            ],
                        )

    return exp_ramsey()


###### Class - based  ######
##### Added by Steph   #####

class ExperimentTemplate():

    fallback_experiment_name = 'Experiment'
    compiled_exp = None

    def __init__(self, qubits, session, measurement_setup, experiment_name=None,
                 signals_list=None, sweeps_dict=None, num_averages=2 ** 10,
                 cal_states=None, datadir=None, do_analysis=True,
                 update_setup=False, save=True, **kwargs):

        self.qubits = qubits
        self.session = session
        self.measurement_setup = measurement_setup

        self.sweeps_dict = sweeps_dict
        self.cal_states = cal_states
        self.num_averages = num_averages
        self.datadir = datadir
        self.do_analysis = do_analysis
        self.update_setup = update_setup
        self.save = save

        self.experiment_name = experiment_name
        if self.experiment_name is None:
            self.experiment_name = self.fallback_experiment_name
        self.create_experiment_label()
        self.signals_list = signals_list
        if self.signals_list is None:
            self.signals_list = ['drive', 'measure', 'acquire']
        self.experiment_signals, self.experiment_signal_uids_qubit_map = \
            self.create_experiment_signals(self.qubits, self.signals_list)
        self.create_experiment(self.experiment_label, signals_list)

    def create_experiment_label(self):
        if len(self.qubits) <= 5:
            qb_names_suffix = f'{"".join([qb.uid for qb in self.qubits])}'
        else:
            qb_names_suffix = f'{len(self.qubits)}qubits'
        self.experiment_label = f'{self.experiment_name}_{qb_names_suffix}'

    @staticmethod
    def signal_name(signal, qubit):
        return f"{signal}_{qubit.uid}"

    @staticmethod
    def create_experiment_signals(qubits, signals_list):
        experiment_signal_uids_qubit_map = {
            qb.uid: [ExperimentTemplate.signal_name(sig, qb)
                     for sig in signals_list] for qb in qubits
        }
        experiment_signals = []
        for qb in qubits:
            for sig in signals_list:
                # assumes signals in signals_list exist in qubits!
                experiment_signals += [ExperimentSignal(f"{sig}_{qb.uid}",
                                                        map_to=qb.signals[sig])]

        return experiment_signals, experiment_signal_uids_qubit_map

    def create_experiment(self, experiment_name, signals_list):
        self.experiment = Experiment(uid=self.experiment_name,
                                     signals=self.experiment_signals)

    def define_experiment(self):
        # Define the experiment acquire looks, sweeps, sections, pulses
        # to be overloaded by children
        pass

    def configure_experiment(self):
        # Set experiment calibration
        # to be overloaded by children
        pass

    def compile_experiment(self):
        if len(self.experiment.sections) == 0:
            self.define_experiment()
        self.configure_experiment()
        self.compiled_exp = self.session.compile(self.experiment)

    def run_experiment(self):
        # create experiment timestamp
        self.timestamp = str(time.strftime("%Y%m%d_%H%M%S"))
        self.results = self.session.run(self.compiled_exp)
        return self.results

    def run_analysis(self):
        # to be overloaded by children
        pass

    @staticmethod
    def update_measurement_setup(qubits, measurement_setup):
        update_setup_calibration_from_qubits(qubits, measurement_setup)

    def save_experiment(self):
        self.savedir = os.path.abspath(os.path.join(
            self.datadir, f'{self.timestamp[:8]}',
            f'{self.timestamp[-6:]}_{self.experiment_label}'))
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)

        # Save all qubit parameters in one json file
        qb_pars_file = os.path.abspath(os.path.join(self.savedir,
                                                    'qubit_parameters.json'))
        with open(qb_pars_file, 'w') as fpo:
            json.dump({qb.uid: qb.parameters.__dict__ for qb in self.qubits},
                      fpo, indent=3)
        # Save individual qubit parameters using QuantumElement.save()
        for qb in self.qubits:
            qb_pars_file = os.path.abspath(os.path.join(
                self.savedir, f'{qb.uid}_parameters.json'))
            qb.save(qb_pars_file)
        # Save Results
        results_file = os.path.abspath(os.path.join(
            self.savedir, f'results.json'))
        self.results.save(results_file)

    def autorun(self):
        if self.compiled_exp is None:
            self.compile_experiment()
        self.results = self.run_experiment()
        if self.do_analysis:
            self.run_analysis()
            if self.update_setup:
                self.update_measurement_setup(self.qubits,
                                              self.measurement_setup)
        if self.save:
            self.save_experiment()
        return self.results

    def add_acquire_rt_loop(self):
        acquire_loop = AcquireLoopRt(
            uid="RT_Acquire_Loop",
            count=self.num_averages,
            averaging_mode=AveragingMode.CYCLIC,
            repetition_mode=RepetitionMode.AUTO,
        )
        self.experiment.add(acquire_loop)
        return acquire_loop

    def create_measure_acquire_sections(self, uid, qubit, play_after=None,
                                        handle_suffix=''):
        handle = f"{self.experiment_name}_{qubit.uid}"
        if len(handle_suffix) > 0:
            handle += f'_{handle_suffix}'
        measure_acquire_section = Section(uid=uid)
        measure_acquire_section.play_after = play_after
        measure_acquire_section.measure(
                measure_signal=self.signal_name('measure', qubit),
                measure_pulse=qt_ops.readout_pulse(qubit),
                handle=handle,
                acquire_signal=self.signal_name('acquire', qubit),
                integration_kernel=qt_ops.readout_pulse(qubit),
                reset_delay=qubit.parameters.user_defined["reset_delay_length"],
            )
        return measure_acquire_section

        # with self.experiment.section(uid=uid, play_after=play_after):
        #     self.experiment.measure(
        #         measure_signal=f"measure_{qubit.uid}",
        #         measure_pulse=qt_ops.readout_pulse(qubit),
        #         handle=handle,
        #         acquire_signal=f"acquire_{qubit.uid}",
        #         integration_kernel=qt_ops.readout_pulse(qubit),
        #         reset_delay=qubit.parameters.user_defined["reset_delay_length"],
        #     )

    def add_cal_states_sections(self, qubit):
        if self.cal_states is None:
            return

        play_after_sections = []
        cal_trace_sections = []
        if 'g' in self.cal_states:
            # Ground state - just a msmt
            g_measure_section = self.create_measure_acquire_sections(
                uid=f"{qubit.uid}_cal_trace_g_meas",
                qubit=qubit,
                handle_suffix='cal_trace_g'
            )
            play_after_sections = [g_measure_section]
            cal_trace_sections += [g_measure_section]
            # self.add_measure_sections(uid=f"{qubit.uid}_cal_trace_g_meas",
            #                           qubit=qubit,
            #                           handle_suffix='cal_trace_g')
        if 'e' in self.cal_states:
            # Excited state - prep pulse + msmt
            e_section = Section(uid=f"{qubit.uid}_cal_trace_e",
                                play_after=play_after_sections)
            e_section.play(
                    signal=self.signal_name('drive', qubit),
                    pulse=qt_ops.quantum_gate(qubit, 'X180_ge'))
            e_measure_section = self.create_measure_acquire_sections(
                    uid=f"{qubit.uid}_cal_trace_e_meas", qubit=qubit,
                play_after=f"{qubit.uid}_cal_trace_e",
                    handle_suffix='cal_trace_e')
            play_after_sections = ([s for s in play_after_sections] +
                                   [e_section, e_measure_section])
            cal_trace_sections += [e_section, e_measure_section]
            # with self.experiment.section(
            #         uid=f"{qubit.uid}_cal_trace_e",
            #         play_after=f"{qubit.uid}_cal_trace_g_meas"):
            #     self.experiment.play(
            #         signal=f"drive_{qubit.uid}",
            #         pulse=qt_ops.quantum_gate(qubit, 'X180_ge'))
            # self.add_measure_sections(uid=f"{qubit.uid}_cal_trace_e_meas",
            #                           qubit=qubit,
            #                           play_after=f"{qubit.uid}_cal_trace_e",
            #                           handle_suffix='cal_trace_e')
        if 'f' in self.cal_states:
            # Excited state - prep pulse + msmt
            f_section = Section(uid=f"{qubit.uid}_cal_trace_f",
                                play_after=play_after_sections)
            # prepare e state
            f_section.play(
                    signal=self.signal_name('drive', qubit),
                    pulse=qt_ops.quantum_gate(qubit, 'X180_ge'))
            # prepare f state
            f_section.play(
                    signal=self.signal_name('drive', qubit),
                    pulse=qt_ops.quantum_gate(qubit, 'X180_ef'))
            measure_section = self.create_measure_acquire_sections(
                    uid=f"{qubit.uid}_cal_trace_f_meas", qubit=qubit,
                    play_after=f"{qubit.uid}_cal_trace_f",
                    handle_suffix='cal_trace_f')
            cal_trace_sections += [f_section, measure_section]
            # with self.experiment.section(
            #         uid=f"{qubit.uid}_cal_trace_f",
            #         play_after=f"{qubit.uid}_cal_trace_g_meas"):
            #     self.experiment.play(
            #         signal=f"drive_{qubit.uid}",
            #         pulse=qt_ops.quantum_gate(qubit, 'X180_ge'))
            #     self.experiment.play(
            #         signal=f"drive_{qubit.uid}",
            #         pulse=qt_ops.quantum_gate(qubit, 'X180_ef'))
            # self.add_measure_sections(uid=f"{qubit.uid}_cal_trace_e_meas",
            #                           qubit=qubit,
            #                           play_after=f"{qubit.uid}_cal_trace_f",
            #                           handle_suffix='cal_trace_e')
        for cal_tr_sec in cal_trace_sections:
            self.experiment.add(cal_tr_sec)



### Single-Qubit Gate Tune-up Experiment classes ###


class SingleQubitGateTuneup(ExperimentTemplate):

    def __init__(self, *args, transition_to_calib='ge', **kwargs):
        self.transition_to_calib = transition_to_calib
        super().__init__(*args, **kwargs)
        if self.cal_states is None:
            self.cal_states = 'gef' if 'f' in self.transition_to_calib else 'ge'

    def play_preparation_pulses(self, qubit):
        if self.transition_to_calib == 'ge':
            return
        elif self.transition_to_calib == 'ef':
            self.experiment.play(
                signal=self.signal_name('drive', qubit),
                pulse=qt_ops.quantum_gate(qubit, 'X180_ge'))
        elif self.transition_to_calib == 'fh':
            self.experiment.play(
                signal=self.signal_name('drive', qubit),
                pulse=qt_ops.quantum_gate(qubit, 'X180_ge'))
            self.experiment.play(
                signal=self.signal_name('drive', qubit),
                pulse=qt_ops.quantum_gate(qubit, 'X180_ef'))
        else:
            raise ValueError(f'Transitions name {self.transition_to_calib} '
                             f'not recognised. Please used one of '
                             f'["ge", "ef", "fh"].')

    # def add_preparation_pulses_to_section(self, section, qubit):
    #     if self.transition_to_calib == 'ge':
    #         return
    #     elif self.transition_to_calib == 'ef':
    #         section.play(
    #             signal=f"drive_{qubit.uid}",
    #             pulse=qt_ops.quantum_gate(qubit, 'X180_ge'))
    #     elif self.transition_to_calib == 'fh':
    #         section.play(
    #             signal=f"drive_{qubit.uid}",
    #             pulse=qt_ops.quantum_gate(qubit, 'X180_ge'))
    #         section.play(
    #             signal=f"drive_{qubit.uid}",
    #             pulse=qt_ops.quantum_gate(qubit, 'X180_ef'))
    #     else:
    #         raise ValueError(f'Transitions name {self.transition_to_calib} '
    #                          f'not recognised. Please used one of '
    #                          f'["ge", "ef", "fh"].')



class AmplitudeRabi(SingleQubitGateTuneup):

    fallback_experiment_name = 'Rabi'
    def define_experiment(self):
        # # define Rabi experiment pulse sequence
        # # outer loop - real-time, cyclic averaging
        # acquire_loop_rt = self.add_acquire_rt_loop()
        # for i, qubit in enumerate(self.qubits):
        #     # create sweep
        #     sweep = Sweep(uid=f"{qubit.uid}_{self.experiment_name}_sweep",
        #                   parameters=[self.sweeps_dict[qubit.uid]])
        #     # create pulses section
        #     excitation_section = Section(
        #         uid=f"{qubit.uid}_excitation", alignment=SectionAlignment.RIGHT)
        #     # preparation pulses: ge if calibrating ef
        #     self.add_preparation_pulses_to_section(excitation_section, qubit)
        #     # pulse to calibrate
        #     x180_pulse = qt_ops.quantum_gate(
        #         qubit, f'X180_{self.transition_to_calib}')
        #     excitation_section.play(signal=f"drive_{qubit.uid}",
        #                             pulse=x180_pulse,
        #                             amplitude=self.sweeps_dict[qubit.uid],)
        #     # create readout + acquire sections
        #     measure_sections = self.create_measure_acquire_sections(
        #         uid=f"{qubit.uid}_readout", qubit=qubit,
        #         play_after=f"{qubit.uid}_excitation")
        #
        #     # add sweep and sections to acquire loop rt
        #     acquire_loop_rt.add(sweep)
        #     sweep.add(excitation_section)
        #     sweep.add(measure_sections)
        #     self.add_cal_states_sections(qubit)

        with self.experiment.acquire_loop_rt(
                uid=f"{self.experiment_name}_shots",
                count=self.num_averages,
                averaging_mode=AveragingMode.CYCLIC,
                acquisition_type=AcquisitionType.INTEGRATION,
        ):
            for i, qubit in enumerate(self.qubits):
                x180_pulse = qt_ops.quantum_gate(
                    qubit, f'X180_{self.transition_to_calib}')
                # inner loop - real time sweep of Rabi amplitudes
                with self.experiment.sweep(
                        uid=f"{qubit.uid}_{self.experiment_name}_sweep",
                        parameter=self.sweeps_dict[qubit.uid]):
                    # qubit drive
                    with self.experiment.section(
                            uid=f"{qubit.uid}_excitation",
                            alignment=SectionAlignment.RIGHT):
                        # X180_ge pulse if calibrating ef
                        self.play_preparation_pulses(qubit)
                        self.experiment.play(
                            signal=self.signal_name('drive', qubit),
                            pulse=x180_pulse,
                            amplitude=self.sweeps_dict[qubit.uid],
                        )
                    # measurement
                    measure_sections = self.create_measure_acquire_sections(
                        uid=f"{qubit.uid}_readout", qubit=qubit,
                        play_after=f"{qubit.uid}_excitation")
                    self.experiment.add(measure_sections)

                # add cal traces
                self.add_cal_states_sections(qubit)



class Ramsey(SingleQubitGateTuneup):

    fallback_experiment_name = 'Ramsey'
    def __init__(self, *args, detuning=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.detuning = detuning
        if self.detuning is None:
            raise ValueError('Please provide detuning.')

    def define_experiment(self):
        with self.experiment.acquire_loop_rt(
                uid=f"{self.experiment_name}_shots",
                count=self.num_averages,
                averaging_mode=AveragingMode.CYCLIC,
                #             repetition_mode=RepetitionMode.AUTO, # needed to be able to have different delays sweeps for qbs
                acquisition_type=AcquisitionType.INTEGRATION,
        ):
            # inner loop - real time sweep of Rabi amplitudes
            with self.experiment.sweep(uid=f"{self.experiment_name}_sweep",
                                       parameter=[self.sweeps_dict[qubit.uid]
                                                  for qubit in self.qubits]):
                for i, qubit in enumerate(self.qubits):
                    # qubit drive
                    with self.experiment.section(
                            uid=f"{qubit.uid}_excitation",
                            alignment=SectionAlignment.RIGHT):
                        # X180_ge pulse if calibrating ef
                        self.play_preparation_pulses(qubit)
                        # Ramsey pulses
                        ramsey_drive_pulse = qt_ops.quantum_gate(
                            qubit,f'X90_{self.transition_to_calib}')
                        self.experiment.play(
                            signal=self.signal_name('drive', qubit),
                            pulse=ramsey_drive_pulse)
                        self.experiment.delay(
                            signal=self.signal_name('drive', qubit),
                            time=self.sweeps_dict[qubit.uid])
                        self.experiment.play(
                            signal=self.signal_name('drive', qubit),
                            pulse=ramsey_drive_pulse)

                    # measurement
                    measure_sections = self.create_measure_acquire_sections(
                        uid=f"{qubit.uid}_readout", qubit=qubit,
                        play_after=f"{qubit.uid}_excitation")
                    self.experiment.add(measure_sections)

                    # add cal traces
                    self.add_cal_states_sections(qubit)

    def configure_experiment(self):
        calib = Calibration()
        for i, qubit in enumerate(self.qubits):
            res_freq = qubit.parameters.resonance_frequency_ef if \
                self.transition_to_calib == 'ef' else \
                qubit.parameters.resonance_frequency_ge
            print(res_freq, self.detuning[qubit.uid],
                   qubit.parameters.drive_lo_frequency)
            freq = res_freq + self.detuning[qubit.uid] - \
                   qubit.parameters.drive_lo_frequency
            calib[self.signal_name("drive", qubit)] = SignalCalibration(
                oscillator=Oscillator(
                    frequency=freq, modulation_type=ModulationType.HARDWARE
                )
            )
        self.experiment.set_calibration(calib)


class QScale(SingleQubitGateTuneup):

    fallback_experiment_name = 'QScale'
    def define_experiment(self):
        with self.experiment.acquire_loop_rt(
                uid=f"{self.experiment_name}_shots",
                count=self.num_averages,
                averaging_mode=AveragingMode.CYCLIC,
                acquisition_type=AcquisitionType.INTEGRATION,
        ):
            for i, qubit in enumerate(self.qubits):
                tn = self.transition_to_calib
                X90_pulse = qt_ops.quantum_gate(qubit, f'X90_{tn}')
                X180_pulse = qt_ops.quantum_gate(qubit, f'X180_{tn}')
                Y180_pulse = qt_ops.quantum_gate(qubit, f'Y180_{tn}')
                Ym180_pulse = qt_ops.quantum_gate(qubit, f'mY180_{tn}')
                pulse_ids = ['xy', 'xx', 'xmy']
                pulses_2nd = [Y180_pulse, X180_pulse, Ym180_pulse]

                swp = self.sweeps_dict[qubit.uid]
                with self.experiment.sweep(
                        uid=f"{qubit.uid}_{self.experiment_name}_sweep",
                        parameter=swp,
                        alignment=SectionAlignment.RIGHT):

                    for i, pulse_2nd in enumerate(pulses_2nd):
                        id = pulse_ids[i]
                        play_after = f'{qubit.uid}_{pulse_ids[i-1]}_section' \
                            if i > 0 else None
                        with self.experiment.section(
                                uid=f'{qubit.uid}_{id}_section',
                                play_after=play_after,
                                alignment=SectionAlignment.RIGHT):
                            # X180_ge pulse if calibrating ef
                            self.play_preparation_pulses(qubit)
                            # qscale pulses
                            self.experiment.play(
                                signal=self.signal_name('drive', qubit),
                                pulse=X90_pulse, pulse_parameters={'beta': swp})
                            self.experiment.play(
                                signal=self.signal_name('drive', qubit),
                                pulse=pulse_2nd, pulse_parameters={'beta': swp})
                        measure_sections = self.create_measure_acquire_sections(
                            uid=f"{qubit.uid}_{id}_section_meas", qubit=qubit,
                            play_after=f"{qubit.uid}_{id}_section")
                        self.experiment.add(measure_sections)

                    # # XX pulses and msmst
                    # with self.experiment.section(
                    #         uid=f'{qubit.uid}_xx_section',
                    #         play_after=f'{qubit.uid}_xy_section',
                    #         alignment=SectionAlignment.RIGHT):
                    #     # X180_ge pulse if calibrating ef
                    #     self.play_preparation_pulses(qubit)
                    #     # qscale pulses
                    #     self.experiment.play(signal=f'drive_{qubit.uid}',
                    #                          pulse=X90_pulse,
                    #                          pulse_parameters={'beta': swp})
                    #     self.experiment.play(signal=f'drive_{qubit.uid}',
                    #                          pulse=X180_pulse,
                    #                          pulse_parameters={'beta': swp})
                    # with self.experiment.section(
                    #         uid=f"{qubit.uid}_xx_section_meas",
                    #         play_after=f"{qubit.uid}_xx_section"):
                    #     self.experiment.measure(
                    #         measure_signal=f"measure_{qubit.uid}",
                    #         measure_pulse=qt_ops.readout_pulse(qubit),
                    #         handle=f"{self.experiment_name}_{qubit.uid}",
                    #         acquire_signal=f"acquire_{qubit.uid}",
                    #         integration_kernel=qt_ops.readout_pulse(qubit),
                    #         reset_delay=qubit.parameters.user_defined[
                    #             "reset_delay_length"],
                    #     )
                    # measure_sections = self.create_measure_acquire_sections(
                    #     uid=f"{qubit.uid}_xx_section_meas", qubit=qubit,
                    #     play_after=f"{qubit.uid}_xx_section")
                    # self.experiment.add(measure_sections)
                    #
                    # # XmY pulses + msmt
                    # with self.experiment.section(
                    #         uid=f'{qubit.uid}_xmy_section',
                    #         play_after=f'{qubit.uid}_xx_section',
                    #         alignment=SectionAlignment.RIGHT):
                    #     # X180_ge pulse if calibrating ef
                    #     self.play_preparation_pulses(qubit)
                    #     # qscale pulses
                    #     self.experiment.play(signal=f'drive_{qubit.uid}',
                    #                          pulse=X90_pulse,
                    #                          pulse_parameters={'beta': swp})
                    #     self.experiment.play(signal=f'drive_{qubit.uid}',
                    #                          pulse=Ym180_pulse,
                    #                          pulse_parameters={'beta': swp})
                    # measure_sections = self.create_measure_acquire_sections(
                    #     uid=f"{qubit.uid}_xx_section_meas", qubit=qubit,
                    #     play_after=f"{qubit.uid}_xx_section")
                    # self.experiment.add(measure_sections)

                # cal traces
                self.add_cal_states_sections(qubit)


class T1(SingleQubitGateTuneup):

    fallback_experiment_name = 'T1'

    def define_experiment(self):
        with self.experiment.acquire_loop_rt(
            uid=f"{self.experiment_name}_shots",
            count=self.num_averages,
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.INTEGRATION,
        ):
            with self.experiment.sweep(
                    uid=f"{self.experiment_name}_sweep",
                    parameter=[self.sweeps_dict[qb.uid] for qb in self.qubits]):
                for i, qubit in enumerate(self.qubits):
                    x180_pulse = qt_ops.quantum_gate(
                        qubit, f'X180_{self.transition_to_calib}')
                    # pi pulse
                    with self.experiment.section(
                            uid=f"{qubit.uid}_excitation",
                            alignment=SectionAlignment.RIGHT):
                        # X180_ge pulse if calibrating ef
                        self.play_preparation_pulses(qubit)
                        self.experiment.play(
                            signal=self.signal_name('drive', qubit),
                            pulse=x180_pulse)
                        self.experiment.delay(
                            signal=self.signal_name('drive', qubit),
                            time=self.sweeps_dict[qubit.uid])

                    # measurement
                    measure_sections = self.create_measure_acquire_sections(
                        uid=f"{qubit.uid}_readout", qubit=qubit,
                        play_after=f"{qubit.uid}_excitation")
                    self.experiment.add(measure_sections)

                    # add cal traces
                    self.add_cal_states_sections(qubit)


class Echo(SingleQubitGateTuneup):

    fallback_experiment_name = 'Echo'
    def __init__(self, *args, detuning=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.detuning = detuning

    def define_experiment(self):
        with self.experiment.acquire_loop_rt(
                uid=f"{self.experiment_name}_shots",
                count=self.num_averages,
                averaging_mode=AveragingMode.CYCLIC,
                acquisition_type=AcquisitionType.INTEGRATION,
        ):
            for i, qubit in enumerate(self.qubits):
                echo_sweep = SweepParameter(
                    uid=f"echo_delays_{qubit.uid}",
                    values=self.sweeps_dict[qubit.uid].values / 2)
                with self.experiment.sweep(
                        uid=f"{qubit.uid}_{self.experiment_name}_sweep",
                        parameter=echo_sweep):
                    with self.experiment.section(
                            uid=f"{qubit.uid}_echo",
                            alignment=SectionAlignment.RIGHT):
                        # X180_ge pulse if calibrating ef
                        self.play_preparation_pulses(qubit)
                        # Echo pulses
                        ramsey_drive_pulse = qt_ops.quantum_gate(
                            qubit, f'X90_{self.transition_to_calib}')
                        echo_drive_pulse = qt_ops.quantum_gate(
                            qubit, f'X180_{self.transition_to_calib}')
                        self.experiment.play(
                            signal=self.signal_name('drive', qubit),
                            pulse=ramsey_drive_pulse)
                        self.experiment.delay(
                            signal=self.signal_name('drive', qubit),
                            time=echo_sweep)
                        self.experiment.play(
                            signal=self.signal_name('drive', qubit),
                            pulse=echo_drive_pulse)
                        self.experiment.delay(
                            signal=self.signal_name('drive', qubit),
                            time=echo_sweep)
                        self.experiment.play(
                            signal=self.signal_name('drive', qubit),
                            pulse=ramsey_drive_pulse)
                    # measurement
                    measure_sections = self.create_measure_acquire_sections(
                        uid=f"{qubit.uid}_readout", qubit=qubit,
                        play_after=f"{qubit.uid}_echo")
                    self.experiment.add(measure_sections)

                # add cal traces
                self.add_cal_states_sections(qubit)

    def configure_experiment(self):
        calib = Calibration()
        for i, qubit in enumerate(self.qubits):
            res_freq = qubit.parameters.resonance_frequency_ef if \
                self.transition_to_calib == 'ef' else \
                qubit.parameters.resonance_frequency_ge
            freq = res_freq + self.detuning[qubit.uid] - \
                   qubit.parameters.drive_lo_frequency
            calib[self.signal_name("drive", qubit)] = SignalCalibration(
                oscillator=Oscillator(
                    frequency=freq, modulation_type=ModulationType.HARDWARE
                )
            )
        self.experiment.set_calibration(calib)
