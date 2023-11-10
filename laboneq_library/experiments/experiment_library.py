from laboneq.dsl.experiment.builtins import *  # noqa: F403
from laboneq.simple import *  # noqa: F403

from . import quantum_operations as qt_ops
from laboneq_library.calibration_helpers import \
    update_setup_calibration_from_qubits
from laboneq.contrib.example_helpers.plotting import plot_helpers as plt_hlp
import time
import pickle
import json
import os
from ruamel.yaml import YAML

ryaml = YAML()


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
                 signals=None, sweeps_dict=None, experiment_metainfo=None,
                 acquisition_metainfo=None, cal_states=None, datadir=None,
                 do_analysis=True, update_setup=False, save=True, **kwargs):

        self.qubits = qubits
        self.session = session
        self.measurement_setup = measurement_setup

        self.sweeps_dict = sweeps_dict
        if self.sweeps_dict is None:
            self.sweeps_dict = {}
        for key, sd in self.sweeps_dict.items():
            if not hasattr(sd, '__iter__'):
                self.sweeps_dict[key] = [sd]
        self.cal_states = cal_states

        self.experiment_metainfo = experiment_metainfo
        if self.experiment_metainfo is None:
            self.experiment_metainfo = {}
        if acquisition_metainfo is None:
            acquisition_metainfo = {}
        self.acquisition_metainfo = dict(count=2 ** 10)
        # overwrite default with user-provided options
        self.acquisition_metainfo.update(acquisition_metainfo)

        self.datadir = datadir
        self.do_analysis = do_analysis
        self.update_setup = update_setup
        self.save = save

        self.experiment_name = experiment_name
        if self.experiment_name is None:
            self.experiment_name = self.fallback_experiment_name
        self.create_experiment_label()
        self.signals = signals
        if self.signals is None:
            self.signals = ['drive', 'measure', 'acquire']
        self.experiment_signals, self.experiment_signal_uids_qubit_map = \
            self.create_experiment_signals(self.qubits, self.signals)
        self.create_experiment()

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
    def create_experiment_signals(qubits, signals):
        experiment_signal_uids_qubit_map = {
            qb.uid: [ExperimentTemplate.signal_name(sig, qb)
                     for sig in signals] for qb in qubits
        }
        experiment_signals = []
        for qb in qubits:
            for sig in signals:
                # assumes signals in signals_list exist in qubits!
                experiment_signals += [ExperimentSignal(f"{sig}_{qb.uid}",
                                                        map_to=qb.signals[sig])]

        return experiment_signals, experiment_signal_uids_qubit_map

    def create_experiment(self):
        self.experiment = Experiment(uid=self.experiment_name,
                                     signals=self.experiment_signals)

    def define_experiment(self):
        # Define the experiment acquire loops, sweeps, sections, pulses
        # to be overloaded by children
        pass

    def configure_experiment(self):
        # Set experiment calibration
        # This method sets the experiment calibration from the
        # qubit calibration of a signal line.
        # To be overloaded by children for overwriting settings.

        cal = Calibration()
        for qubit in self.qubits:
            for sig in self.signals:  # 'drive', 'flux', 'measure', 'acquire'
                cal[self.signal_name(sig, qubit)] = \
                    qubit.calibration()[qubit.signals[sig]]
        self.experiment.set_calibration(cal)

    def compile_experiment(self):
        if len(self.experiment.sections) == 0:
            self.define_experiment()
        calib = self.experiment.get_calibration()
        if all([cv is None for cv in calib.values()]):
            self.configure_experiment()
        self.compiled_exp = self.session.compile(self.experiment)

    def run_experiment(self):
        if self.compiled_exp is None:
            self.compile_experiment()
        if self.save:
            self.create_timestamp_savedir()
        self.results = self.session.run(self.compiled_exp)
        return self.results

    def run_analysis(self):
        # to be overloaded by children
        pass

    @staticmethod
    def update_measurement_setup(qubits, measurement_setup):
        update_setup_calibration_from_qubits(qubits, measurement_setup)

    def create_timestamp_savedir(self):
        # create experiment timestamp
        self.timestamp = str(time.strftime("%Y%m%d_%H%M%S"))
        # create experiment savedir
        self.savedir = os.path.abspath(os.path.join(
            self.datadir, f'{self.timestamp[:8]}',
            f'{self.timestamp[-6:]}_{self.experiment_label}'))
        # create the savedir
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)

    def save_experiment(self):
        if not hasattr(self, 'savedir'):
            self.create_timestamp_savedir()

        # # Save Results
        # results_file = os.path.abspath(os.path.join(
        #     self.savedir, f'results.json'))
        # self.results.save(results_file)

        # Save acquired results
        filename = os.path.abspath(os.path.join(
            self.savedir, 'acquired_results.p'))
        with open(filename, 'wb') as f:
            pickle.dump(self.results.acquired_results, f)

        # qubit_parameters = {qb.uid: qb.parameters.__dict__ for qb in self.qubits}
        # # Save all qubit parameters in one yaml file
        # qb_pars_file = os.path.abspath(os.path.join(self.savedir,
        #                                             'qubit_parameters.yaml'))
        # with open(qb_pars_file, "w") as file:
        #     ryaml.dump(qubit_parameters, file)

        # # Save all qubit parameters in one json file
        # qb_pars_file = os.path.abspath(os.path.join(self.savedir,
        #                                             'qubit_parameters.json'))
        # with open(qb_pars_file, 'w') as fpo:
        #     json.dump(qubit_parameters, fpo, indent=3)
        #
        # # Save individual qubit parameters using QuantumElement.save()
        # for qb in self.qubits:
        #     qb_pars_file = os.path.abspath(os.path.join(
        #         self.savedir, f'{qb.uid}_parameters.json'))
        #     qb.save(qb_pars_file)

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

    def add_acquire_rt_loop(self, section_container=None):
        self.acquire_loop = AcquireLoopRt(
            uid="RT_Acquire_Loop", **self.acquisition_metainfo)
        if section_container is None:
            self.experiment.add(self.acquire_loop)
        else:
            section_container.add(self.acquire_loop)

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
            integration_length=qubit.parameters.readout_integration_length,
            reset_delay=qubit.parameters.user_defined["reset_delay_length"],
        )
        return measure_acquire_section

    def add_cal_states_sections(self, qubit, section_container=None):
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
        for cal_tr_sec in cal_trace_sections:
            if section_container is None:
                self.acquire_loop.add(cal_tr_sec)
            else:
                section_container.add(cal_tr_sec)


class ResonatorSpectroscopy(ExperimentTemplate):
    fallback_experiment_name = 'ResonatorSpectroscopy'

    def __init__(self, *args, **kwargs):
        kwargs['signals'] = kwargs.pop('signals', ['measure', 'acquire'])

        acquisition_metainfo_user = kwargs.pop('acquisition_metainfo', dict())
        acquisition_metainfo = dict(acquisition_type=AcquisitionType.SPECTROSCOPY)
        acquisition_metainfo.update(acquisition_metainfo_user)
        kwargs['acquisition_metainfo'] = acquisition_metainfo

        experiment_metainfo = kwargs.get('experiment_metainfo', dict())
        self.nt_swp_par = experiment_metainfo.get('neartime_sweep_parameter',
                                                  'frequency')
        super().__init__(*args, **kwargs)

    def define_experiment(self):
        self.experiment.sections = []
        cw = self.experiment_metainfo.get('continuous_wave', True)
        for qubit in self.qubits:
            nt_sweep = None
            qb_sweep = self.sweeps_dict[qubit.uid]
            if len(qb_sweep) > 1:
                nt_sweep_func = qb_sweep[1]
                nt_sweep = Sweep(
                    uid=f"neartime_{self.nt_swp_par}_sweep_{qubit.uid}",
                    parameters=[nt_sweep_func])
                self.experiment.add(nt_sweep)
                if self.nt_swp_par == 'voltage':
                    ntsf = self.experiment_metainfo.get(
                        'neartime_callback_function', None)
                    if ntsf is None:
                        raise ValueError(
                            "Please provide the neartime callback function for "
                            "the voltage sweep in "
                            "experiment_metainfo['neartime_sweep_prameter'].")
                    # all near-time callback functions have the format
                    # func(session, sweep_param_value, qubit)
                    nt_sweep.call(ntsf, voltage=nt_sweep_func, qubit=qubit)

            # define real-time loop
            self.add_acquire_rt_loop(nt_sweep)
            inner_freq_sweep = qb_sweep[0]
            sweep_inner = Sweep(uid=f"resonator_frequency_inner_{qubit.uid}",
                                parameters=[inner_freq_sweep])
            measure_acquire_section = Section(uid=f'measure_acquire_{qubit.uid}')
            if cw:
                measure_acquire_section.measure(
                    measure_signal=None,
                    handle=f"{self.experiment_name}_{qubit.uid}",
                    acquire_signal=self.signal_name('acquire', qubit),
                    integration_length=qubit.parameters.readout_integration_length,
                    # reset_delay=qubit.parameters.user_defined["reset_delay_length"],
                )
            else:
                measure_acquire_section.measure(
                    measure_signal=self.signal_name('measure', qubit),
                    measure_pulse=qt_ops.readout_pulse(qubit),
                    handle=f"{self.experiment_name}_{qubit.uid}",
                    acquire_signal=self.signal_name('acquire', qubit),
                    integration_kernel=qt_ops.readout_pulse(qubit),
                    integration_length=qubit.parameters.readout_integration_length,
                    # reset_delay=qubit.parameters.user_defined["reset_delay_length"],
                )
            reserve_sec = Section(uid=f"delay_{qubit.uid}", length=1e-6)
            # holdoff time after signal acquisition
            reserve_sec.reserve(signal=f"measure_{qubit.uid}")
            reserve_sec.reserve(signal=f"acquire_{qubit.uid}")

            sweep_inner.add(measure_acquire_section)
            sweep_inner.add(reserve_sec)
            self.acquire_loop.add(sweep_inner)

    def configure_experiment(self):
        super().configure_experiment()

        # configure sweep
        for qubit in self.qubits:
            qb_sweep = self.sweeps_dict[qubit.uid]
            local_oscillator = None
            ro_amplitude = None
            if len(qb_sweep) > 1:
                if self.nt_swp_par == 'amplitude':
                    ro_amplitude = qb_sweep[1]
                elif self.nt_swp_par == 'frequency':
                    local_oscillator = Oscillator(frequency=qb_sweep[1])

            freq_swp = qb_sweep[0]
            if all(freq_swp.values > 1e9):
                # sweep values are passed as qubit resonance frequencies:
                # subtract readout lo freq to sweep if
                freq_swp = SweepParameter(
                    f'if_freq_{qubit.uid}',
                    values=qb_sweep[0].values -
                           qubit.parameters.readout_lo_frequency)

            cal_measure = self.experiment.signals[
                self.signal_name('measure', qubit)].calibration
            cal_measure.oscillator = Oscillator(
                frequency=freq_swp, modulation_type=ModulationType.HARDWARE)
            cal_measure.local_oscillator = local_oscillator
            cal_measure.amplitude = ro_amplitude

            cal_acquire = self.experiment.signals[
                self.signal_name('acquire', qubit)].calibration
            cal_acquire.local_oscillator = local_oscillator


class QubitSpectroscopy(ExperimentTemplate):
    fallback_experiment_name = 'QubitSpectroscopy'

    def __init__(self, *args, **kwargs):
        experiment_metainfo = kwargs.get('experiment_metainfo', dict())
        self.nt_swp_par = experiment_metainfo.get('neartime_sweep_parameter',
                                                  'frequency')
        super().__init__(*args, **kwargs)

    def define_experiment(self):
        self.experiment.sections = []
        cw = self.experiment_metainfo.get('continuous_wave', True)
        for qubit in self.qubits:
            nt_sweep = None
            qb_sweep = self.sweeps_dict[qubit.uid]
            if len(qb_sweep) > 1:
                nt_sweep_func = qb_sweep[1]
                nt_sweep = Sweep(
                    uid=f"neartime_{self.nt_swp_par}_sweep_{qubit.uid}",
                    parameters=[nt_sweep_func])
                self.experiment.add(nt_sweep)
                if self.nt_swp_par == 'voltage':
                    ntsf = self.experiment_metainfo.get(
                        'neartime_callback_function', None)
                    if ntsf is None:
                        raise ValueError(
                            "Please provide the neartime callback function for "
                            "the voltage sweep in "
                            "experiment_metainfo['neartime_sweep_prameter'].")
                    # all near-time callback functions have the format
                    # func(session, sweep_param_value, qubit)
                    nt_sweep.call(ntsf, voltage=nt_sweep_func, qubit=qubit)

            # define real-time loop
            self.add_acquire_rt_loop(nt_sweep)
            sweep = Sweep(uid=f"ge_frequency_sweep_{qubit.uid}",
                          parameters=[self.sweeps_dict[qubit.uid][0]])

            if not cw:
                excitation_section = Section(uid=f"{qubit.uid}_excitation")
                spec_pulse = pulse_library.const(
                    uid=f"spectroscopy_pulse_{qubit.uid}",
                    length=qubit.parameters.user_defined["spec_length"],
                    amplitude=qubit.parameters.user_defined["spec_amplitude"],
                    can_compress=True
                )
                excitation_section.play(
                    signal=self.signal_name('drive', qubit),
                    pulse=spec_pulse)
                sweep.add(excitation_section)

            measure_sections = self.create_measure_acquire_sections(
                uid=f"{qubit.uid}_readout", qubit=qubit,
                play_after=f"{qubit.uid}_excitation" if not cw else None)
            sweep.add(measure_sections)
            self.acquire_loop.add(sweep)

    def configure_experiment(self):
        super().configure_experiment()
        for qubit in self.qubits:
            qb_sweep = self.sweeps_dict[qubit.uid]
            local_oscillator = None
            drive_amplitude = None
            if len(qb_sweep) > 1:
                if self.nt_swp_par == 'amplitude':
                    drive_amplitude = qb_sweep[1]
                elif self.nt_swp_par == 'frequency':
                    local_oscillator = Oscillator(frequency=qb_sweep[1])

            freq_swp = self.sweeps_dict[qubit.uid][0]
            if all(freq_swp.values > 1e9):
                # sweep values are passed as qubit resonance frequencies:
                # subtract lo freq to sweep if
                freq_swp = SweepParameter(
                    f'if_freq_{qubit.uid}',
                    values=self.sweeps_dict[qubit.uid][0].values -
                           qubit.parameters.drive_lo_frequency)

            cal_drive = self.experiment.signals[
                self.signal_name('drive', qubit)].calibration
            cal_drive.oscillator = Oscillator(
                frequency=freq_swp, modulation_type=ModulationType.HARDWARE)
            cal_drive.local_oscillator = local_oscillator
            cal_drive.amplitude = drive_amplitude
            


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

    def add_preparation_pulses_to_section(self, section, qubit):
        if self.transition_to_calib == 'ge':
            return
        elif self.transition_to_calib == 'ef':
            section.play(
                signal=self.signal_name('drive', qubit),
                pulse=qt_ops.quantum_gate(qubit, 'X180_ge'))
        elif self.transition_to_calib == 'fh':
            section.play(
                signal=self.signal_name('drive', qubit),
                pulse=qt_ops.quantum_gate(qubit, 'X180_ge'))
            section.play(
                signal=self.signal_name('drive', qubit),
                pulse=qt_ops.quantum_gate(qubit, 'X180_ef'))
        else:
            raise ValueError(f'Transitions name {self.transition_to_calib} '
                             f'not recognised. Please used one of '
                             f'["ge", "ef", "fh"].')

    def run_analysis(self):
        plt_hlp.plot_results(self.results, savedir=self.savedir)


class AmplitudeRabi(SingleQubitGateTuneup):
    fallback_experiment_name = 'Rabi'

    def define_experiment(self):
        # define Rabi experiment pulse sequence
        # outer loop - real-time, cyclic averaging
        self.add_acquire_rt_loop()
        for i, qubit in enumerate(self.qubits):
            # create sweep
            sweep = Sweep(uid=f"{qubit.uid}_{self.experiment_name}_sweep",
                          parameters=[self.sweeps_dict[qubit.uid][0]])
            # create pulses section
            excitation_section = Section(
                uid=f"{qubit.uid}_excitation", alignment=SectionAlignment.RIGHT)
            # preparation pulses: ge if calibrating ef
            self.add_preparation_pulses_to_section(excitation_section, qubit)
            # pulse to calibrate
            x180_pulse = qt_ops.quantum_gate(
                qubit, f'X180_{self.transition_to_calib}')
            excitation_section.play(signal=f"drive_{qubit.uid}",
                                    pulse=x180_pulse,
                                    amplitude=self.sweeps_dict[qubit.uid][0], )
            # create readout + acquire sections
            measure_sections = self.create_measure_acquire_sections(
                uid=f"{qubit.uid}_readout", qubit=qubit,
                play_after=f"{qubit.uid}_excitation")

            # add sweep and sections to acquire loop rt
            self.acquire_loop.add(sweep)
            sweep.add(excitation_section)
            sweep.add(measure_sections)
            self.add_cal_states_sections(qubit)


class Ramsey(SingleQubitGateTuneup):
    fallback_experiment_name = 'Ramsey'

    def define_experiment(self):
        self.add_acquire_rt_loop()
        # create joint sweep for all qubits
        sweep = Sweep(uid=f"{self.experiment_name}_sweep",
                      parameters=[self.sweeps_dict[qubit.uid][0]
                                  for qubit in self.qubits])
        self.acquire_loop.add(sweep)
        for i, qubit in enumerate(self.qubits):
            # create pulses section
            excitation_section = Section(
                uid=f"{qubit.uid}_excitation", alignment=SectionAlignment.RIGHT)
            # preparation pulses: ge if calibrating ef
            self.add_preparation_pulses_to_section(excitation_section, qubit)
            # Ramsey pulses
            ramsey_drive_pulse = qt_ops.quantum_gate(
                qubit, f'X90_{self.transition_to_calib}')
            excitation_section.play(
                signal=self.signal_name('drive', qubit),
                pulse=ramsey_drive_pulse)
            excitation_section.delay(
                signal=self.signal_name('drive', qubit),
                time=self.sweeps_dict[qubit.uid][0])
            excitation_section.play(
                signal=self.signal_name('drive', qubit),
                pulse=ramsey_drive_pulse)

            # create readout + acquire sections
            measure_sections = self.create_measure_acquire_sections(
                uid=f"{qubit.uid}_readout", qubit=qubit,
                play_after=f"{qubit.uid}_excitation")

            # add sweep and sections to acquire loop rt
            sweep.add(excitation_section)
            sweep.add(measure_sections)
            self.add_cal_states_sections(qubit)

    def configure_experiment(self):
        super().configure_experiment()
        detuning = self.experiment_metainfo.get('detuning')
        if detuning is None:
            raise ValueError('Please provide detuning in experiment_metainfo.')
        for i, qubit in enumerate(self.qubits):
            res_freq = qubit.parameters.resonance_frequency_ef if \
                self.transition_to_calib == 'ef' else \
                qubit.parameters.resonance_frequency_ge
            freq = res_freq + detuning[qubit.uid] - \
                   qubit.parameters.drive_lo_frequency
            cal_drive = self.experiment.signals[
                self.signal_name('drive', qubit)].calibration
            cal_drive.oscillator = Oscillator(
                frequency=freq, modulation_type=ModulationType.HARDWARE)


class QScale(SingleQubitGateTuneup):
    fallback_experiment_name = 'QScale'

    def define_experiment(self):
        self.add_acquire_rt_loop()
        for i, qubit in enumerate(self.qubits):
            tn = self.transition_to_calib
            X90_pulse = qt_ops.quantum_gate(qubit, f'X90_{tn}')
            X180_pulse = qt_ops.quantum_gate(qubit, f'X180_{tn}')
            Y180_pulse = qt_ops.quantum_gate(qubit, f'Y180_{tn}')
            Ym180_pulse = qt_ops.quantum_gate(qubit, f'mY180_{tn}')
            pulse_ids = ['xy', 'xx', 'xmy']
            pulses_2nd = [Y180_pulse, X180_pulse, Ym180_pulse]

            swp = self.sweeps_dict[qubit.uid][0]
            # create sweep
            sweep = Sweep(uid=f"{qubit.uid}_{self.experiment_name}_sweep",
                          parameters=[swp])
            self.acquire_loop.add(sweep)
            # create pulses sections
            for i, pulse_2nd in enumerate(pulses_2nd):
                id = pulse_ids[i]
                play_after = f"{qubit.uid}_{pulse_ids[i - 1]}_section_meas" \
                    if i > 0 else None

                excitation_section = Section(uid=f'{qubit.uid}_{id}_section',
                                             play_after=play_after,
                                             alignment=SectionAlignment.RIGHT)
                # preparation pulses: ge if calibrating ef
                self.add_preparation_pulses_to_section(excitation_section, qubit)
                # qscale pulses
                excitation_section.play(
                    signal=self.signal_name('drive', qubit),
                    pulse=X90_pulse, pulse_parameters={'beta': swp})
                excitation_section.play(
                    signal=self.signal_name('drive', qubit),
                    pulse=pulse_2nd, pulse_parameters={'beta': swp})

                # create readout + acquire sections
                measure_sections = self.create_measure_acquire_sections(
                    uid=f"{qubit.uid}_{id}_section_meas", qubit=qubit,
                    play_after=f"{qubit.uid}_{id}_section")

                # Add sections to sweep
                sweep.add(excitation_section)
                sweep.add(measure_sections)

            self.add_cal_states_sections(qubit)


class T1(SingleQubitGateTuneup):
    fallback_experiment_name = 'T1'

    def define_experiment(self):
        self.add_acquire_rt_loop()
        # create joint sweep for all qubits
        sweep = Sweep(uid=f"{self.experiment_name}_sweep",
                      parameters=[self.sweeps_dict[qubit.uid][0]
                                  for qubit in self.qubits])
        self.acquire_loop.add(sweep)
        for i, qubit in enumerate(self.qubits):
            # create pulses section
            excitation_section = Section(
                uid=f"{qubit.uid}_excitation", alignment=SectionAlignment.RIGHT)
            # preparation pulses: ge if calibrating ef
            self.add_preparation_pulses_to_section(excitation_section, qubit)
            x180_pulse = qt_ops.quantum_gate(
                qubit, f'X180_{self.transition_to_calib}')
            excitation_section.play(
                signal=self.signal_name('drive', qubit),
                pulse=x180_pulse)
            excitation_section.delay(
                signal=self.signal_name('drive', qubit),
                time=self.sweeps_dict[qubit.uid][0])
            # create readout + acquire sections
            measure_sections = self.create_measure_acquire_sections(
                uid=f"{qubit.uid}_readout", qubit=qubit,
                play_after=f"{qubit.uid}_excitation")

            # add sweep and sections to acquire loop rt
            sweep.add(excitation_section)
            sweep.add(measure_sections)
            self.add_cal_states_sections(qubit)


class Echo(SingleQubitGateTuneup):
    fallback_experiment_name = 'Echo'

    def define_experiment(self):
        self.add_acquire_rt_loop()
        # create joint sweep for all qubits
        sweep_list = [SweepParameter(
            uid=f"echo_delays_{qubit.uid}",
            values=self.sweeps_dict[qubit.uid][0].values / 2)
            for qubit in self.qubits]
        sweep = Sweep(uid=f"{self.experiment_name}_sweep",
                      parameters=sweep_list)
        self.acquire_loop.add(sweep)
        for i, qubit in enumerate(self.qubits):
            # create pulses section
            excitation_section = Section(
                uid=f"{qubit.uid}_excitation", alignment=SectionAlignment.RIGHT)
            # preparation pulses: ge if calibrating ef
            self.add_preparation_pulses_to_section(excitation_section, qubit)
            # Echo pulses
            ramsey_drive_pulse = qt_ops.quantum_gate(
                qubit, f'X90_{self.transition_to_calib}')
            echo_drive_pulse = qt_ops.quantum_gate(
                qubit, f'X180_{self.transition_to_calib}')
            excitation_section.play(
                signal=self.signal_name('drive', qubit),
                pulse=ramsey_drive_pulse)
            excitation_section.delay(
                signal=self.signal_name('drive', qubit),
                time=sweep_list[i])
            excitation_section.play(
                signal=self.signal_name('drive', qubit),
                pulse=echo_drive_pulse)
            excitation_section.delay(
                signal=self.signal_name('drive', qubit),
                time=sweep_list[i])
            excitation_section.play(
                signal=self.signal_name('drive', qubit),
                pulse=ramsey_drive_pulse)
            # create readout + acquire sections
            measure_sections = self.create_measure_acquire_sections(
                uid=f"{qubit.uid}_readout", qubit=qubit,
                play_after=f"{qubit.uid}_excitation")

            # add sweep and sections to acquire loop rt
            sweep.add(excitation_section)
            sweep.add(measure_sections)
            self.add_cal_states_sections(qubit)

    def configure_experiment(self):
        super().configure_experiment()
        detuning = self.experiment_metainfo.get('detuning')
        if detuning is None:
            raise ValueError('Please provide detuning in experiment_metainfo.')

        calib = Calibration()
        for i, qubit in enumerate(self.qubits):
            res_freq = qubit.parameters.resonance_frequency_ef if \
                self.transition_to_calib == 'ef' else \
                qubit.parameters.resonance_frequency_ge
            freq = res_freq + detuning[qubit.uid] - \
                   qubit.parameters.drive_lo_frequency
            cal_drive = self.experiment.signals[
                self.signal_name('drive', qubit)].calibration
            cal_drive.oscillator = Oscillator(
                frequency=freq, modulation_type=ModulationType.HARDWARE)
