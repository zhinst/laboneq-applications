import json
import os
import time
import pickle
import numpy as np
from copy import deepcopy
import uncertainties as unc
import matplotlib.pyplot as plt
from ruamel.yaml import YAML

ryaml = YAML()

import traceback
import logging

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("experiment_library")

from . import quantum_operations as qt_ops
from laboneq.analysis import fitting as fit_mods
from laboneq_library import calibration_helpers as calib_hlp
from laboneq_library.analysis import analysis_helpers as ana_hlp
from laboneq.contrib.example_helpers.plotting import plot_helpers as plt_hlp
from laboneq.dsl.experiment.builtins import *  # noqa: F403
from laboneq.simple import *  # noqa: F403


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
        num_averages=2 ** 10,
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
        num_averages=2 ** 10,
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
                            integration_kernel=qt_ops.integration_kernel(qubit),
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
                            integration_kernel=qt_ops.integration_kernel(qubit),
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
                            integration_kernel=qt_ops.integration_kernel(qubit),
                            reset_delay=qubit.parameters.user_defined[
                                "reset_delay_length"
                            ],
                        )

    return exp_ramsey()


###### Class - based  ######
##### Added by Steph   #####

class ExperimentTemplate():
    fallback_experiment_name = "Experiment"
    savedir = None
    timestamp = None
    compiled_exp = None
    fit_results = None
    new_qubit_parameters = None

    def __init__(self, qubits, session, measurement_setup, experiment_name=None,
                 signals=None, sweep_parameters_dict=None, experiment_metainfo=None,
                 acquisition_metainfo=None, cal_states=None, datadir=None,
                 do_analysis=True, analysis_metainfo=None, save=True,
                 update=False, run=False, **kwargs):

        self.qubits = qubits
        self.session = session
        self.measurement_setup = measurement_setup

        self.sweep_parameters_dict = deepcopy(sweep_parameters_dict)
        if self.sweep_parameters_dict is None:
            self.sweep_parameters_dict = {}
        for key, sd in self.sweep_parameters_dict.items():
            if not hasattr(sd, "__iter__"):
                self.sweep_parameters_dict[key] = [sd]
        self.cal_states = cal_states

        self.experiment_metainfo = experiment_metainfo
        if self.experiment_metainfo is None:
            self.experiment_metainfo = {}
        if acquisition_metainfo is None:
            acquisition_metainfo = {}
        self.acquisition_metainfo = dict(count=2 ** 12)
        # overwrite default with user-provided options
        self.acquisition_metainfo.update(acquisition_metainfo)

        self.datadir = datadir
        self.do_analysis = do_analysis
        self.analysis_metainfo = analysis_metainfo
        if self.analysis_metainfo is None:
            self.analysis_metainfo = {}
        self.update = update
        if self.update and self.analysis_metainfo.get('do_fitting', False):
            log.warning("update is True but "
                        "analysis_metainfo['do_fitting'] is False. Qubit "
                        "parameters will not be updated.")
        self.save = save

        self.experiment_name = experiment_name
        if self.experiment_name is None:
            self.experiment_name = self.fallback_experiment_name
        self.create_experiment_label()
        self.signals = signals
        if self.signals is None:
            self.signals = ["drive", "measure", "acquire"]
        self.experiment_signals, self.experiment_signal_uids_qubit_map = \
            self.create_experiment_signals(self.qubits, self.signals)
        self.create_experiment()

        self.run = run
        if self.run:
            self.autorun()

    def create_experiment_label(self):
        if len(self.qubits) <= 5:
            qb_names_suffix = f'{"".join([qb.uid for qb in self.qubits])}'
        else:
            qb_names_suffix = f"{len(self.qubits)}qubits"
        self.experiment_label = f"{self.experiment_name}_{qb_names_suffix}"

    @staticmethod
    def signal_name(signal, qubit):
        return f"{signal}_{qubit.uid}"

    @staticmethod
    def create_experiment_signals(qubits, signals):
        experiment_signal_uids_qubit_map = {
            qb.uid: [ExperimentTemplate.signal_name(sig, qb) for sig in signals]
            for qb in qubits
        }
        experiment_signals = []
        for qb in qubits:
            for sig in signals:
                # assumes signals in signals_list exist in qubits!
                experiment_signals += [
                    ExperimentSignal(f"{sig}_{qb.uid}", map_to=qb.signals[sig])
                ]

        return experiment_signals, experiment_signal_uids_qubit_map

    def create_experiment(self):
        self.experiment = Experiment(
            uid=self.experiment_name, signals=self.experiment_signals
        )

    def define_experiment(self):
        # Define the experiment acquire loops, sweeps, sections, pulses
        # To be overridden by children
        self.sweeps_dict = {}
        for qubit in self.qubits:
            self.sweeps_dict[qubit.uid] = []
            for i, swp in enumerate(self.sweep_parameters_dict[qubit.uid]):
                self.sweeps_dict[qubit.uid] += [Sweep(
                    uid=f'{qubit.uid}_{self.experiment_name}_sweep_{i}',
                    parameters=[swp]
                )]

    def configure_experiment(self):
        """
        Set experiment calibration.

        This method sets the experiment calibration from the qubit
        calibration of a signal line. To be overridden by children for
        overwriting settings.

        """

        expcal = Calibration()
        for qubit in self.qubits:
            qbcal = qubit.calibration()
            for sig in self.signals:  # 'drive', 'drive_ef', 'flux', 'measure', 'acquire'
                expcal[self.signal_name(sig, qubit)] = qbcal[qubit.signals[sig]]
        self.experiment.set_calibration(expcal)

    def compile_experiment(self):
        if len(self.experiment.sections) == 0:
            self.define_experiment()
        calib = self.experiment.get_calibration()
        if all([cv is None for cv in calib.values()]):
            self.configure_experiment()
        self.compiled_exp = self.session.compile(self.experiment)

    def run_experiment(self):
        if self.savedir is None:
            self.create_timestamp_savedir()
        if self.compiled_exp is None:
            self.compile_experiment()
        self.results = self.session.run(self.compiled_exp)
        return self.results

    def analyse_experiment(self):
        # to be overridden by children
        pass

    @staticmethod
    def update_measurement_setup(qubits, measurement_setup):
        calib_hlp.update_measurement_setup_from_qubits(qubits, measurement_setup)

    def update_qubit_parameters(self):
        pass

    def update_entire_setup(self):
        self.update_qubit_parameters()
        self.update_measurement_setup(self.qubits, self.measurement_setup)

    def create_timestamp_savedir(self):
        # create experiment timestamp
        self.timestamp = str(time.strftime("%Y%m%d_%H%M%S"))
        # create experiment savedir
        self.savedir = os.path.abspath(
            os.path.join(
                self.datadir,
                f"{self.timestamp[:8]}",
                f"{self.timestamp[-6:]}_{self.experiment_label}",
            )
        )
        # create the savedir
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)

    def save_experiment(self):
        if self.savedir is None:
            self.create_timestamp_savedir()

        # Save Results
        results_file = os.path.abspath(os.path.join(
            self.savedir, f'{self.timestamp}_results.json'))
        try:
            self.results.save(results_file)
        except Exception as e:
            log.warning(f'Could not save all the results: {e}')

        # Save acquired results
        filename = os.path.abspath(os.path.join(
            self.savedir, f"{self.timestamp}_acquired_results.p")
        )
        with open(filename, "wb") as f:
            pickle.dump(self.results.acquired_results, f)

        # Save the measurement setup
        filename = os.path.abspath(os.path.join(
            self.savedir, f"{self.timestamp}_measurement_setup.json")
        )
        self.measurement_setup.save(filename)

    def save_figure(self, fig, qubit, figure_name=None):
        if self.savedir is None:
            self.create_timestamp_savedir()
        fig_name = self.analysis_metainfo.get("figure_name", figure_name)
        if fig_name is None:
            fig_name = f"{self.timestamp}_{self.experiment_name}_{qubit.uid}"
        fig.savefig(self.savedir + f"\\{fig_name}.png",
                    bbox_inches="tight", dpi=600)

    def save_fit_results(self):
        if self.fit_results is not None:
            # Save fit results into a json file
            fit_res_file = os.path.abspath(os.path.join(
                self.savedir, f"{self.timestamp}_fit_results.json")
            )
            fit_results_to_save = {}
            for qbuid, fit_res in self.fit_results.items():
                if isinstance(fit_res, dict):
                    fit_results_to_save[qbuid] = {}
                    for k, fr in fit_res.items():
                        fit_results_to_save[qbuid][k] = \
                            ana_hlp.flatten_lmfit_modelresult(fr)
                else:
                    fit_results_to_save[qbuid] = \
                        ana_hlp.flatten_lmfit_modelresult(fit_res)
            with open(fit_res_file, "w") as file:
                json.dump(fit_results_to_save, file, indent=2)
            # Save fit results into a pickle file
            filename = os.path.abspath(os.path.join(
                self.savedir, f"{self.timestamp}_fit_results.p")
            )
            with open(filename, "wb") as f:
                pickle.dump(self.fit_results, f)

    def autorun(self):
        try:
            if self.compiled_exp is None:
                self.compile_experiment()
            self.run_experiment()
            if self.save:
                self.save_experiment()
            if self.do_analysis:
                self.analyse_experiment()
            if self.update:
                self.update_entire_setup()
            return self.results
        except Exception:
            log.error("Unhandled error during experiment!")
            log.error(traceback.format_exc())

    def create_acquire_rt_loop(self):
        self.acquire_loop = AcquireLoopRt(
            uid="RT_Acquire_Loop", **self.acquisition_metainfo
        )

    def create_measure_acquire_sections(self, uid, qubit, play_after=None,
                                        handle_suffix='', integration_kernel=None):
        handle = f"{self.experiment_name}_{qubit.uid}"
        if len(handle_suffix) > 0:
            handle += f"_{handle_suffix}"

        ro_pulse = qt_ops.readout_pulse(qubit)
        if not hasattr(self, 'integration_kernel'):
            # ensure the integration_kernel is created only once to avoid
            # serialisation errors
            self.integration_kernel = pulse_library.const(
                uid=f"integration_kernel_{qubit.uid}",
                length=qubit.parameters.readout_integration_length,
                amplitude=1,
            )
        if integration_kernel is None:
            integration_kernel = self.integration_kernel
        measure_acquire_section = Section(uid=uid)
        measure_acquire_section.play_after = play_after
        measure_acquire_section.measure(
            measure_signal=self.signal_name("measure", qubit),
            measure_pulse=ro_pulse,
            handle=handle,
            acquire_signal=self.signal_name("acquire", qubit),
            integration_kernel=integration_kernel,
            integration_length=qubit.parameters.readout_integration_length,
            reset_delay=qubit.parameters.user_defined["reset_delay_length"],
        )
        return measure_acquire_section

    def add_cal_states_sections(self, qubit, section_container=None):
        if self.cal_states is None:
            return

        play_after_sections = []
        cal_trace_sections = []
        if "g" in self.cal_states:
            # Ground state - just a msmt
            g_measure_section = self.create_measure_acquire_sections(
                uid=f"{qubit.uid}_cal_trace_g_meas",
                qubit=qubit,
                handle_suffix="cal_trace_g",
            )
            play_after_sections = [g_measure_section]
            cal_trace_sections += [g_measure_section]
        if "e" in self.cal_states:
            # Excited state - prep pulse + msmt
            e_section = Section(
                uid=f"{qubit.uid}_cal_trace_e",
                play_after=play_after_sections,
            )
            e_section.play(
                signal=self.signal_name("drive", qubit),
                pulse=qt_ops.quantum_gate(qubit, "X180_ge",
                                          uid=f"{qubit.uid}_cal_trace_e"
                                          ),
            )
            e_measure_section = self.create_measure_acquire_sections(
                uid=f"{qubit.uid}_cal_trace_e_meas",
                qubit=qubit,
                play_after=f"{qubit.uid}_cal_trace_e",
                handle_suffix="cal_trace_e",
            )
            play_after_sections = [s for s in play_after_sections] + [
                e_section,
                e_measure_section,
            ]
            cal_trace_sections += [e_section, e_measure_section]
        if "f" in self.cal_states:
            # Excited state - prep pulse + msmt
            # prepare e state
            e_section = Section(
                uid=f"{qubit.uid}_cal_trace_f_e",
                play_after=play_after_sections,
                on_system_grid=True,
            )
            e_section.play(
                signal=self.signal_name("drive", qubit),
                pulse=qt_ops.quantum_gate(qubit, "X180_ge",
                                          uid=f"{qubit.uid}_cal_trace_f_e"),
            )
            # prepare f state
            f_section = Section(
                uid=f"{qubit.uid}_cal_trace_f_f",
                play_after=play_after_sections + [e_section],
                on_system_grid=True,
            )
            f_section.play(
                signal=self.signal_name("drive_ef", qubit),
                pulse=qt_ops.quantum_gate(qubit, "X180_ef",
                                          uid=f"{qubit.uid}_cal_trace_f_f"),
            )
            measure_section = self.create_measure_acquire_sections(
                uid=f"{qubit.uid}_cal_trace_f_meas",
                qubit=qubit,
                play_after=f"{qubit.uid}_cal_trace_f_f",
                handle_suffix="cal_trace_f",
            )
            cal_trace_sections += [e_section, f_section, measure_section]
        for cal_tr_sec in cal_trace_sections:
            if section_container is None:
                self.acquire_loop.add(cal_tr_sec)
            else:
                section_container.add(cal_tr_sec)


class ResonatorSpectroscopy(ExperimentTemplate):
    fallback_experiment_name = 'ResonatorSpectroscopy'

    def __init__(self, *args, **kwargs):
        kwargs["signals"] = kwargs.pop("signals", ["measure", "acquire"])

        acquisition_metainfo_user = kwargs.pop("acquisition_metainfo", dict())
        acquisition_metainfo = dict(acquisition_type=AcquisitionType.SPECTROSCOPY)
        acquisition_metainfo.update(acquisition_metainfo_user)
        kwargs["acquisition_metainfo"] = acquisition_metainfo

        experiment_metainfo = kwargs.get("experiment_metainfo", dict())
        self.nt_swp_par = experiment_metainfo.get("neartime_sweep_parameter",
                                                  "frequency")
        self.pulsed = experiment_metainfo.get("pulsed", False)
        run = kwargs.pop("run", False)  # instantiate base without running exp
        kwargs["run"] = False
        super().__init__(*args, **kwargs)

        # Add suffix to experiment name
        if self.nt_swp_par != "frequency":
            self.experiment_name += f"{self.nt_swp_par[0].upper()}{self.nt_swp_par[1:]}Sweep"
            self.create_experiment_label()

        for qubit in self.qubits:
            freq_swp = self.sweep_parameters_dict[qubit.uid][0]
            if all(freq_swp.values > 1e9):
                # sweep values are passed as qubit resonance frequencies:
                # subtract lo freq to sweep if freq
                if_freq_swp = SweepParameter(
                    f"if_freq_{qubit.uid}",
                    values=freq_swp.values - qubit.parameters.readout_lo_frequency,
                    axis_name=freq_swp.axis_name,
                    driven_by=[freq_swp]
                )
                self.sweep_parameters_dict[qubit.uid][0] = if_freq_swp

        self.run = run
        if self.run:
            self.autorun()

    def define_experiment(self):
        self.experiment.sections = []
        self.create_acquire_rt_loop()
        for qubit in self.qubits:
            ro_pulse_amp = qubit.parameters.user_defined['readout_amplitude']
            qb_sweep_pars = self.sweep_parameters_dict[qubit.uid]
            nt_sweep = None
            if len(qb_sweep_pars) > 1:
                nt_sweep_par = qb_sweep_pars[1]
                nt_sweep = Sweep(
                    uid=f"neartime_{self.nt_swp_par}_sweep_{qubit.uid}",
                    parameters=[nt_sweep_par])
                self.experiment.add(nt_sweep)
                if self.nt_swp_par == "voltage":
                    ntsf = self.experiment_metainfo.get(
                        "neartime_callback_function", None)
                    if ntsf is None:
                        raise ValueError(
                            "Please provide the neartime callback function for "
                            "the voltage sweep in "
                            "experiment_metainfo['neartime_sweep_prameter'].")
                    # all near-time callback functions have the format
                    # func(session, sweep_param_value, qubit)
                    nt_sweep.call(ntsf, voltage=nt_sweep_par, qubit=qubit)
                elif self.nt_swp_par == 'amplitude':
                    ro_pulse_amp = 1
                # add real-time loop to nt_sweep
                nt_sweep.add(self.acquire_loop)
            else:
                self.experiment.add(self.acquire_loop)

            inner_freq_sweep = qb_sweep_pars[0]
            sweep_inner = Sweep(uid=f"resonator_frequency_inner_{qubit.uid}",
                                parameters=[inner_freq_sweep])
            measure_acquire_section = Section(uid=f'measure_acquire_{qubit.uid}')
            if self.pulsed:
                ro_pulse = pulse_library.const(
                    length=qubit.parameters.user_defined["readout_length"],
                    amplitude=ro_pulse_amp)
                integration_kernel = pulse_library.const(
                    uid=f"integration_kernel_{qubit.uid}",
                    length=qubit.parameters.readout_integration_length,
                    amplitude=1,
                )
                measure_acquire_section.measure(
                    measure_signal=self.signal_name("measure", qubit),
                    measure_pulse=ro_pulse,
                    handle=f"{self.experiment_name}_{qubit.uid}",
                    acquire_signal=self.signal_name("acquire", qubit),
                    integration_kernel=integration_kernel,
                    integration_length=qubit.parameters.readout_integration_length,
                    reset_delay=qubit.parameters.user_defined["reset_delay_length"],
                )
                sweep_inner.add(measure_acquire_section)
            else:
                measure_acquire_section.measure(
                    measure_signal=None,
                    handle=f"{self.experiment_name}_{qubit.uid}",
                    acquire_signal=self.signal_name("acquire", qubit),
                    integration_length=qubit.parameters.readout_integration_length,
                    reset_delay=qubit.parameters.user_defined["reset_delay_length"],
                )
                # why is the reserve_sec needed for cw but not for pulsed?
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
            qb_sweep = self.sweep_parameters_dict[qubit.uid]
            local_oscillator = None
            ro_amplitude = None
            if len(qb_sweep) > 1:
                if self.nt_swp_par == 'amplitude':  # and not self.pulsed:
                    ro_amplitude = qb_sweep[1]
                elif self.nt_swp_par == "frequency":
                    local_oscillator = Oscillator(frequency=qb_sweep[1])

            freq_swp = qb_sweep[0]
            cal_measure = self.experiment.signals[
                self.signal_name("measure", qubit)].calibration
            cal_measure.oscillator = Oscillator(
                frequency=freq_swp, modulation_type=ModulationType.HARDWARE)
            if local_oscillator is not None:
                cal_measure.local_oscillator = local_oscillator
            cal_acquire = self.experiment.signals[
                self.signal_name("acquire", qubit)].calibration
            cal_acquire.local_oscillator = local_oscillator
            if ro_amplitude is not None:
                cal_measure.amplitude = ro_amplitude

    def analyse_experiment(self):
        ts = self.timestamp if self.timestamp is not None else ''
        self.new_qubit_parameters = {}
        self.fit_results = {}
        freq_filter = self.analysis_metainfo.get('frequency_filter_for_fit', {})
        find_peaks = self.analysis_metainfo.get('find_peaks', {})
        for qubit in self.qubits:
            self.new_qubit_parameters[qubit.uid] = {}
            # get frequency filter of qubit
            ff_qb = freq_filter.get(qubit.uid, None)
            # decide whether to extract peaks or dips for qubit
            fp_qb = find_peaks.get(qubit.uid, False)
            take_extremum = np.argmax if fp_qb else np.argmin
            # extract data
            handle = f"{self.experiment_name}_{qubit.uid}"
            data_mag = abs(self.results.get_data(handle))
            res_axis = self.results.get_axis(handle)
            if self.nt_swp_par == "frequency":
                data_mag = np.array([data for data in data_mag]).flatten()
                if len(res_axis) > 1:
                    outer = self.results.get_axis(handle)[0]
                    inner = self.results.get_axis(handle)[1]
                    freqs = np.array([out + inner for out in outer]).flatten()
                else:
                    freqs = self.results.get_axis(handle)[0] + \
                            qubit.parameters.readout_lo_frequency

                data_to_search = data_mag if ff_qb is None else data_mag[ff_qb(freqs)]
                freqs_to_search = freqs if ff_qb is None else freqs[ff_qb(freqs)]
                f0 = freqs_to_search[take_extremum(data_to_search)]
                d0 = data_to_search[take_extremum(data_to_search)]
                self.new_qubit_parameters[qubit.uid]["readout_resonator_frequency"] = f0

                # plot data
                fig, ax = plt.subplots()
                ax.plot(freqs / 1e9, data_mag)
                ax.plot(f0 / 1e9, d0, 'ro')
                textstr = f'Extracted readout-resonator frequency: {f0 / 1e9:.4f} GHz'
                textstr += (f'\nCurrent readout-resonator frequency: '
                            f'{qubit.parameters.readout_resonator_frequency / 1e9:.4f} GHz')
                ax.text(0, -0.15, textstr, ha='left', va='top',
                        transform=ax.transAxes)
                ax.set_xlabel(self.results.get_axis_name(handle)[0])
                ax.set_ylabel("Signal Magnitude (a.u.)")
            else:
                # 2D plot of results
                nt_sweep_par_vals = self.results.get_axis(handle)[0]
                nt_sweep_par_name = self.results.get_axis_name(handle)[0]
                freqs = self.results.get_axis(handle)[1] + \
                        qubit.parameters.readout_lo_frequency
                freqs_axis_name = self.results.get_axis_name(handle)[1]
                data_mag = abs(self.results.get_data(handle))

                X, Y = np.meshgrid(freqs / 1e9, nt_sweep_par_vals)
                fig, ax = plt.subplots(constrained_layout=True)

                CS = ax.contourf(X, Y, data_mag, levels=100, cmap="magma")
                ax.set_title(f"{handle}")
                ax.set_xlabel(freqs_axis_name)
                ax.set_ylabel(nt_sweep_par_name)
                cbar = fig.colorbar(CS)
                cbar.set_label("Signal Magnitude (a.u.)")

                if self.nt_swp_par == 'voltage':
                    # 1D plot of the qubit frequency vs voltage
                    if ff_qb is None:
                        freqs_dips = freqs[take_extremum(data_mag, axis=1)]
                    else:
                        mask = ff_qb(freqs)
                        freqs_dips = freqs[mask][take_extremum(
                            data_mag[:, mask], axis=1)]
                    # plot
                    ax.plot(freqs_dips / 1e9, nt_sweep_par_vals, 'ow')
                    # figure out whether voltages vs freqs is convex or concave
                    take_extremum_fit, scf = (np.argmax, 1) if (
                        ana_hlp.is_data_convex(nt_sweep_par_vals, freqs_dips)) \
                        else (np.argmin, -1)
                    # optimal parking parameters at the extremum of
                    # voltages vs frequencies
                    f0 = freqs_dips[take_extremum_fit(freqs_dips)]
                    V0 = nt_sweep_par_vals[take_extremum_fit(freqs_dips)]
                    self.new_qubit_parameters[qubit.uid].update({
                        "readout_resonator_frequency": f0,
                        "dc_voltage_parking": V0
                    })

                    if self.analysis_metainfo.get('do_fitting', True):
                        # fit frequency vs voltage and take the optimal parking
                        # parameters from fit
                        data_to_fit = freqs_dips
                        swpts_to_fit = nt_sweep_par_vals
                        freqs_guess, phase_guess = ana_hlp.find_oscillation_frequency_and_phase(
                            data_to_fit, swpts_to_fit)
                        param_hints = self.analysis_metainfo.get(
                            'param_hints', {
                                'frequency': {'value': 2 * np.pi * freqs_guess,
                                              'min': 0},
                                'phase': {'value': phase_guess},
                                'amplitude': {'value': abs(max(data_to_fit) -
                                                           min(data_to_fit)) / 2,
                                              'min': 0},
                                'offset': {'value': np.mean(data_to_fit)}
                            })
                        fit_res = ana_hlp.fit_data_lmfit(
                            fit_mods.oscillatory, swpts_to_fit, data_to_fit,
                            param_hints=param_hints)
                        self.fit_results[qubit.uid] = fit_res

                        # extract USS and LSS voltages and frequencies
                        freq_fit = unc.ufloat(fit_res.params['frequency'].value,
                                              fit_res.params['frequency'].stderr)
                        phase_fit = unc.ufloat(fit_res.params['phase'].value,
                                               fit_res.params['phase'].stderr)
                        voltages_uss, voltages_lss, _, _ = ana_hlp.get_pi_pi2_xvalues_on_cos(
                            swpts_to_fit, freq_fit, phase_fit)
                        v_uss_values = np.array([vuss.nominal_value for vuss in voltages_uss])
                        v_lss_values = np.array([vlss.nominal_value for vlss in voltages_lss])
                        freqs_uss = fit_res.model.func(v_uss_values, **fit_res.best_values)
                        freqs_lss = fit_res.model.func(v_lss_values, **fit_res.best_values)

                        # plot fit
                        swpts_fine = np.linspace(swpts_to_fit[0],
                                                 swpts_to_fit[-1], 501)
                        ax.plot(fit_res.model.func(
                            swpts_fine, **fit_res.best_values) / 1e9,
                                swpts_fine, 'w-')
                        line_uss, = ax.plot(freqs_uss / 1e9, v_uss_values, 'bo')
                        line_lss, = ax.plot(freqs_lss / 1e9, v_lss_values, 'go')

                        # extract parking values, show them on plot and save
                        # them in self.new_qubit_parameters
                        self.new_qubit_parameters[qubit.uid].update({
                            "readout_resonator_frequency": {},
                            "dc_voltage_parking": {}
                        })
                        if len(v_uss_values) > 0:
                            uss_idx = np.argsort(abs(v_uss_values))[0]
                            v_uss, f_uss = voltages_uss[uss_idx], freqs_uss[uss_idx]
                            textstr = f"Smallest USS voltage:\n" + \
                                      f"{v_uss.nominal_value:.4f} V $\\pm$ {v_uss.std_dev:.4f} V"
                            textstr += f"\nParking frequency:\n{f_uss / 1e9:.4f} GHz"
                            ax.text(1, -0.15, textstr, ha='right', va='top',
                                    c=line_uss.get_c(), transform=ax.transAxes)
                            self.new_qubit_parameters[qubit.uid][
                                "readout_resonator_frequency"]['uss'] = v_uss.nominal_value
                            self.new_qubit_parameters[qubit.uid][
                                "dc_voltage_parking"]['uss'] = f_uss
                        if len(v_lss_values) > 0:
                            lss_idx = np.argsort(abs(v_lss_values))[0]
                            v_lss, f_lss = voltages_lss[lss_idx], freqs_lss[lss_idx]
                            textstr = f"Smallest LSS voltage:\n" + \
                                      f"{v_lss.nominal_value:.4f} V $\\pm$ {v_lss.std_dev:.4f} V"
                            textstr += f"\nParking frequency:\n{f_lss / 1e9:.4f} GHz"
                            ax.text(0, -0.15, textstr, ha='left', va='top',
                                    c=line_lss.get_c(), transform=ax.transAxes)
                            self.new_qubit_parameters[qubit.uid][
                                "readout_resonator_frequency"]['lss'] = v_lss.nominal_value
                            self.new_qubit_parameters[qubit.uid][
                                "dc_voltage_parking"]['lss'] = f_lss

            ax.set_title(f'{ts}_{handle}')
            # save figures and results
            if self.save:
                # Save the figure
                self.save_figure(fig, qubit)
                if len(self.fit_results) > 0:
                    # Save fit results
                    self.save_fit_results()
            if self.analysis_metainfo.get("show_figures", False):
                plt.show()
            plt.close(fig)

    def update_qubit_parameters(self):
        for qubit in self.qubits:
            new_qb_pars = self.new_qubit_parameters[qubit.uid]
            if len(new_qb_pars) == 4:
                # both uss and lss found
                raise ValueError('Both upper and lower sweep spots were found. '
                                 'Unclear which one to set. Please update '
                                 'qubit parameters manually.')
            qubit.parameters.readout_resonator_frequency = new_qb_pars[
                "readout_resonator_frequency"]
            if "dc_voltage_parking" in new_qb_pars:
                qubit.parameters.user_defined["dc_voltage_parking"] = new_qb_pars[
                    "dc_voltage_parking"]


class QubitSpectroscopy(ExperimentTemplate):
    fallback_experiment_name = "QubitSpectroscopy"

    def __init__(self, *args, **kwargs):
        experiment_metainfo = kwargs.get('experiment_metainfo', dict())
        self.nt_swp_par = experiment_metainfo.get('neartime_sweep_parameter',
                                                  'frequency')
        self.pulsed = experiment_metainfo.get('pulsed', True)
        run = kwargs.pop('run', False)  # instantiate base without running exp
        kwargs['run'] = False
        super().__init__(*args, **kwargs)

        # Add suffix to experiment name
        if self.nt_swp_par != "frequency":
            self.experiment_name += f"{self.nt_swp_par[0].upper()}{self.nt_swp_par[1:]}Sweep"
            self.create_experiment_label()

        for qubit in self.qubits:
            freq_swp = self.sweep_parameters_dict[qubit.uid][0]
            if all(freq_swp.values > 1e9):
                # sweep values are passed as qubit resonance frequencies:
                # subtract lo freq to sweep if freq
                if_freq_swp = SweepParameter(
                    f'if_freq_{qubit.uid}',
                    values=freq_swp.values - qubit.parameters.drive_lo_frequency,
                    axis_name=freq_swp.axis_name,
                    driven_by=[freq_swp])
                self.sweep_parameters_dict[qubit.uid][0] = if_freq_swp

        self.run = run
        if self.run:
            self.autorun()

    def define_experiment(self):
        self.experiment.sections = []
        self.create_acquire_rt_loop()

        if len(self.sweep_parameters_dict[self.qubits[0].uid]) > 1:
            # 2D sweep
            nt_sweep_pars = [self.sweep_parameters_dict[qubit.uid][1]
                             for qubit in self.qubits]
            if self.nt_swp_par == 'voltage':
                ntsf = self.experiment_metainfo.get(
                    'neartime_callback_function', None)
                if ntsf is None:
                    raise ValueError(
                        "Please provide the neartime callback function for "
                        "the voltage sweep in "
                        "experiment_metainfo['neartime_sweep_prameter'].")
                # all near-time callback functions have the format
                # func(session, sweep_param_value)
                voltages_array = np.array([ntsp.values for ntsp in nt_sweep_pars]).T
                slots_array = np.array([qubit.parameters.user_defined['dc_slot'] - 1
                                        for qubit in self.qubits])
                slots_array = slots_array[np.newaxis, :]
                slots_array = np.repeat(slots_array, voltages_array.shape[0], axis=0)
                sweep_array = np.zeros((voltages_array.shape[0],
                                        2 * voltages_array.shape[1]))
                sweep_array[:, 0::2] = slots_array
                sweep_array[:, 1::2] = voltages_array
                nt_sweep_par = SweepParameter(uid=f"dc_voltage_sweep",
                                              values=sweep_array)
                nt_sweep = Sweep(
                    uid=f"neartime_{self.nt_swp_par}_sweep",
                    parameters=[nt_sweep_par])
                nt_sweep.call(ntsf, voltage=nt_sweep_par)
            else:
                nt_sweep = Sweep(
                    uid=f"neartime_{self.nt_swp_par}_sweep",
                    parameters=nt_sweep_pars)
            self.experiment.add(nt_sweep)
        else:
            self.experiment.add(self.acquire_loop)

        # create joint frequency sweep for all qubits
        sweep_pars_freq = [self.sweep_parameters_dict[qubit.uid][0]
                           for qubit in self.qubits]
        sweep_freq = Sweep(
            uid=f"{self.experiment_name}_sweep",
            parameters=sweep_pars_freq)
        self.acquire_loop.add(sweep_freq)
        for i, qubit in enumerate(self.qubits):
            spec_pulse_amp = qubit.parameters.user_defined["spec_amplitude"]
            if self.nt_swp_par == 'amplitude':
                spec_pulse_amp = nt_sweep_pars[i]

            integration_kernel = None
            if self.pulsed:
                excitation_section = Section(uid=f"{qubit.uid}_excitation")
                spec_pulse = pulse_library.const(
                    uid=f"spectroscopy_pulse_{qubit.uid}",
                    length=qubit.parameters.user_defined["spec_length"],
                    amplitude=spec_pulse_amp,
                    can_compress=True  # fails without this!
                )
                integration_kernel = pulse_library.const(
                    uid=f"integration_kernel_{qubit.uid}",
                    length=qubit.parameters.readout_integration_length,
                    amplitude=qubit.parameters.user_defined['readout_amplitude'],
                )
                excitation_section.play(
                    signal=self.signal_name("drive", qubit),
                    pulse=spec_pulse
                )
                sweep_freq.add(excitation_section)

            measure_sections = self.create_measure_acquire_sections(
                uid=f"{qubit.uid}_readout",
                qubit=qubit,
                integration_kernel=integration_kernel,
                play_after=f"{qubit.uid}_excitation" if self.pulsed else None)
            sweep_freq.add(measure_sections)


        # for qubit in self.qubits:
        #     spec_pulse_amp = qubit.parameters.user_defined["spec_amplitude"]
        #     qb_sweep_pars = self.sweep_parameters_dict[qubit.uid]
        #     if len(qb_sweep_pars) > 1:
        #         nt_sweep_par = qb_sweep_pars[1]
        #         nt_sweep = Sweep(
        #             uid=f"neartime_{self.nt_swp_par}_sweep_{qubit.uid}",
        #             parameters=[nt_sweep_par])
        #         self.experiment.add(nt_sweep)
        #         if self.nt_swp_par == 'voltage':
        #             ntsf = self.experiment_metainfo.get(
        #                 'neartime_callback_function', None)
        #             if ntsf is None:
        #                 raise ValueError(
        #                     "Please provide the neartime callback function for "
        #                     "the voltage sweep in "
        #                     "experiment_metainfo['neartime_sweep_prameter'].")
        #             # all near-time callback functions have the format
        #             # func(session, sweep_param_value, qubit)
        #             nt_sweep.call(ntsf, voltage=nt_sweep_par, qubit=qubit)
        #         elif self.nt_swp_par == 'amplitude':
        #             spec_pulse_amp = nt_sweep_par
        #         # add real-time loop to nt_sweep
        #         nt_sweep.add(self.acquire_loop)
        #     else:
        #         self.experiment.add(self.acquire_loop)
        #
        #     freq_sweep = Sweep(uid=f"frequency_sweep_{qubit.uid}",
        #                        parameters=[self.sweep_parameters_dict[qubit.uid][0]])
        #     integration_kernel = None
        #     if self.pulsed:
        #         excitation_section = Section(uid=f"{qubit.uid}_excitation")
        #         spec_pulse = pulse_library.const(
        #             uid=f"spectroscopy_pulse_{qubit.uid}",
        #             length=qubit.parameters.user_defined["spec_length"],
        #             amplitude=spec_pulse_amp,
        #             can_compress=True  # fails without this!
        #         )
        #         integration_kernel = pulse_library.const(
        #             uid=f"integration_kernel_{qubit.uid}",
        #             length=qubit.parameters.readout_integration_length,
        #             amplitude=qubit.parameters.user_defined['readout_amplitude'],
        #         )
        #         excitation_section.play(
        #             signal=self.signal_name("drive", qubit), pulse=spec_pulse
        #         )
        #         freq_sweep.add(excitation_section)
        #
        #     measure_sections = self.create_measure_acquire_sections(
        #         uid=f"{qubit.uid}_readout",
        #         qubit=qubit,
        #         integration_kernel=integration_kernel,
        #         play_after=f"{qubit.uid}_excitation" if self.pulsed else None)
        #
        #     freq_sweep.add(measure_sections)
        #     self.acquire_loop.add(freq_sweep)

    def configure_experiment(self):
        super().configure_experiment()
        for qubit in self.qubits:
            qb_sweep = self.sweep_parameters_dict[qubit.uid]
            local_oscillator = None
            drive_amplitude = None
            if len(qb_sweep) > 1:
                if self.nt_swp_par == 'amplitude' and not self.pulsed:
                    drive_amplitude = qb_sweep[1]
                elif self.nt_swp_par == 'frequency':
                    local_oscillator = Oscillator(frequency=qb_sweep[1])

            freq_swp = self.sweep_parameters_dict[qubit.uid][0]
            cal_drive = self.experiment.signals[
                self.signal_name('drive', qubit)].calibration
            cal_drive.oscillator = Oscillator(
                frequency=freq_swp, modulation_type=ModulationType.HARDWARE)
            if local_oscillator is not None:
                cal_drive.local_oscillator = local_oscillator
            if drive_amplitude is not None:
                cal_drive.amplitude = drive_amplitude

    def analyse_experiment(self):
        ts = self.timestamp if self.timestamp is not None else ''
        self.new_qubit_parameters = {}
        self.fit_results = {}
        freq_filter = self.analysis_metainfo.get('frequency_filter_for_fit', {})
        find_peaks = self.analysis_metainfo.get('find_peaks', {})
        for qubit in self.qubits:
            # get frequency filter of qubit
            ff_qb = freq_filter.get(qubit.uid, None)
            # extract data
            handle = f"{self.experiment_name}_{qubit.uid}"
            data_mag = abs(self.results.get_data(handle))
            res_axis = self.results.get_axis(handle)
            if self.nt_swp_par == 'frequency':
                data_mag = np.array([data for data in data_mag]).flatten()
                if len(res_axis) > 1:
                    outer = self.results.get_axis(handle)[0]
                    inner = self.results.get_axis(handle)[1]
                    freqs = np.array([out + inner for out in outer]).flatten()
                else:
                    freqs = self.results.get_axis(handle)[0] + \
                            qubit.parameters.drive_lo_frequency

                # plot data
                fig, ax = plt.subplots()
                ax.plot(freqs / 1e9, data_mag, 'o')
                ax.set_xlabel(self.results.get_axis_name(handle)[0])
                ax.set_ylabel("Signal Magnitude (a.u.)")

                if self.analysis_metainfo.get('do_fitting', True):
                    data_to_fit = data_mag if ff_qb is None else data_mag[ff_qb(freqs)]
                    freqs_to_fit = freqs if ff_qb is None else freqs[ff_qb(freqs)]
                    # fit data
                    param_hints = self.analysis_metainfo.get('param_hints')
                    if param_hints is None:
                        width_guess = 50e3
                        # fit with guess values for a peak
                        param_hints = {
                            'amplitude': {'value': np.max(data_to_fit)*width_guess},
                            'position': {'value': freqs_to_fit[np.argmax(data_to_fit)]},
                            'width': {'value': width_guess},
                            'offset': {'value': 0}
                         }
                        fit_res_peak = ana_hlp.fit_data_lmfit(
                            fit_mods.lorentzian, freqs_to_fit, data_to_fit,
                            param_hints=param_hints)
                        # fit with guess values for a dip
                        param_hints['amplitude']['value'] *= -1
                        param_hints['position']['value'] = freqs_to_fit[np.argmin(data_to_fit)]
                        fit_res_dip = ana_hlp.fit_data_lmfit(
                            fit_mods.lorentzian, freqs_to_fit, data_to_fit,
                            param_hints=param_hints)
                        # determine whether there is a peak or a dip: compare
                        # the distance between the value at the fitted peak/dip
                        # to the mean of the data_mag array: the larger distance
                        # is the true spectroscopy signal
                        dpeak = abs(fit_res_peak.model.func(
                            fit_res_peak.best_values['position'],
                            **fit_res_peak.best_values) - np.mean(data_to_fit))
                        ddip = abs(fit_res_dip.model.func(
                            fit_res_dip.best_values['position'],
                            **fit_res_dip.best_values) - np.mean(data_to_fit))
                        fit_res = fit_res_peak if dpeak > ddip else fit_res_dip
                    else:
                        # do what the user asked
                        fit_res = ana_hlp.fit_data_lmfit(
                            fit_mods.lorentzian, freqs_to_fit, data_to_fit,
                            param_hints=param_hints)
                    self.fit_results[qubit.uid] = fit_res
                    fqb = fit_res.params['position'].value
                    fqb_err = fit_res.params['position'].stderr
                    self.new_qubit_parameters[qubit.uid] = {
                        "resonance_frequency_ge": fqb}

                    # plot fit
                    freqs_fine = np.linspace(freqs_to_fit[0], freqs_to_fit[-1], 501)
                    ax.plot(freqs_fine / 1e9, fit_res.model.func(
                        freqs_fine, **fit_res.best_values), 'r-')
                    textstr = (f'Extracted qubit frequency: {fqb / 1e9:.4f} GHz '
                               f'$\\pm$ {fqb_err / 1e9:.4f} GHz')
                    textstr += (f'\nCurrent qubit frequency: '
                                f'{qubit.parameters.resonance_frequency_ge / 1e9:.4f} GHz')
                    ax.text(0, -0.15, textstr, ha='left', va='top',
                            transform=ax.transAxes)
            else:
                # 2D plot of results
                nt_sweep_par_vals = self.results.get_axis(handle)[0]
                nt_sweep_par_name = self.results.get_axis_name(handle)[0]
                freqs = self.results.get_axis(handle)[1] + \
                        qubit.parameters.drive_lo_frequency
                freqs_axis_name = self.results.get_axis_name(handle)[1]
                data_mag = abs(self.results.get_data(handle))

                X, Y = np.meshgrid(freqs / 1e9, nt_sweep_par_vals)
                fig, ax = plt.subplots(constrained_layout=True)

                CS = ax.contourf(X, Y, data_mag, levels=100, cmap="magma")
                ax.set_title(f"{handle}")
                ax.set_xlabel(freqs_axis_name)
                ax.set_ylabel(nt_sweep_par_name)
                cbar = fig.colorbar(CS)
                cbar.set_label("Signal Magnitude (a.u.)")

                if self.nt_swp_par == 'voltage':
                    # 1D plot of the qubit frequency vs voltage
                    # decide whether to extract peaks or dips for qubit
                    fp_qb = find_peaks.get(qubit.uid, True)
                    take_extremum = np.argmax if fp_qb else np.argmin
                    if ff_qb is None:
                        freqs_peaks = freqs[take_extremum(data_mag, axis=1)]
                    else:
                        mask = ff_qb(freqs)
                        print(len(freqs), len(freqs[mask]))
                        freqs_peaks = freqs[mask][take_extremum(
                            data_mag[:, mask], axis=1)]
                    # plot
                    ax.plot(freqs_peaks / 1e9, nt_sweep_par_vals, 'ow')
                    # figure out whether voltages vs freqs is convex or concave
                    take_extremum_fit, scf = (np.argmax, 1) if (
                        ana_hlp.is_data_convex(nt_sweep_par_vals, freqs_peaks)) \
                        else (np.argmin, -1)
                    # optimal parking parameters at the extremum of
                    # voltages vs frequencies
                    f0 = freqs_peaks[take_extremum_fit(freqs_peaks)]
                    V0 = nt_sweep_par_vals[take_extremum_fit(freqs_peaks)]
                    self.new_qubit_parameters[qubit.uid] = {
                        "readout_resonator_frequency": f0,
                        "dc_voltage_parking": V0
                    }

                    if self.analysis_metainfo.get('do_fitting', True):
                        # fit frequency vs voltage and take the optimal parking
                        # parameters from fit
                        # fit_func = lambda x, V0, f0, fv: f0 - fv * (x - V0) ** 2
                        param_hints = {
                            'V0': {'value': V0},
                            'f0': {'value': f0},
                            'fv': {'value': scf * (max(freqs_peaks) - min(freqs_peaks))},
                        }
                        fit_res = ana_hlp.fit_data_lmfit(
                            fit_mods.transmon_voltage_dependence_quadratic,
                            nt_sweep_par_vals, freqs_peaks, param_hints=param_hints)
                        self.fit_results[qubit.uid] = fit_res
                        self.new_qubit_parameters[qubit.uid] = {
                            "resonance_frequency_ge": fit_res.best_values['f0'],
                            "dc_voltage_parking": fit_res.best_values['V0']
                        }
                        # plot fit
                        ntpval_fine = np.linspace(nt_sweep_par_vals[0],
                                                  nt_sweep_par_vals[-1], 501)
                        ax.plot(fit_res.model.func(
                            ntpval_fine, **fit_res.best_values) / 1e9,
                                ntpval_fine, 'w-')
                        f0, f0err = fit_res.best_values['f0'], fit_res.params['f0'].stderr
                        V0, V0err = fit_res.best_values['V0'], fit_res.params['V0'].stderr
                        ax.plot(f0 / 1e9, V0, 'sC2',
                                markersize=plt.rcParams['lines.markersize'] + 1)
                        textstr = f"Parking voltage: {V0:.4f} $\\pm$ {V0err:.4f} V"
                        textstr += f"\nParking frequency: {f0 / 1e9:.4f} $\\pm$ {f0err / 1e9:.4f} GHz"
                        ax.text(0, -0.15, textstr, ha='left', va='top',
                                transform=ax.transAxes)

            ax.set_title(f'{ts}_{handle}')
            # save figures and results
            if self.save:
                # Save the figure
                self.save_figure(fig, qubit)
                if len(self.fit_results) > 0:
                    # Save fit results
                    self.save_fit_results()
            if self.analysis_metainfo.get("show_figures", False):
                plt.show()
            plt.close(fig)

    def update_qubit_parameters(self):
        for qubit in self.qubits:
            new_qb_pars = self.new_qubit_parameters[qubit.uid]
            qubit.parameters.resonance_frequency_ge = new_qb_pars[
                "resonance_frequency_ge"]
            if "dc_voltage_parking" in new_qb_pars:
                qubit.parameters.user_defined["dc_voltage_parking"] = new_qb_pars[
                    "dc_voltage_parking"]


### Single-Qubit Gate Tune-up Experiment classes ###


class SingleQubitGateTuneup(ExperimentTemplate):
    def __init__(self, *args, signals=None, transition_to_calib="ge", **kwargs):
        self.transition_to_calib = transition_to_calib
        # suffix of the drive signal
        self.drive_signal_suffix = "_ef" if self.transition_to_calib == "ef" else ''

        cal_states = kwargs.get("cal_states", None)
        if cal_states is None:
            cal_states = "gef" if 'f' in self.transition_to_calib else "ge"
        kwargs["cal_states"] = cal_states

        if signals is None:
            signals = ["drive", "measure", "acquire"]
        if 'f' in self.transition_to_calib and "drive_ef" not in signals:
            signals += ["drive_ef"]

        run = kwargs.pop("run", False)  # instantiate base without running exp
        kwargs["run"] = False
        super().__init__(*args, signals=signals, **kwargs)

        self.experiment_name += f"_{self.transition_to_calib}"
        self.create_experiment_label()

        self.run = run
        if self.run:
            self.autorun()

    def add_preparation_pulses_to_section(self, section, qubit):
        if self.transition_to_calib == "ge":
            return
        elif self.transition_to_calib == "ef":
            uid = f"{qubit.uid}_prep_e"
            section.play(
                signal=self.signal_name("drive", qubit),
                pulse=qt_ops.quantum_gate(qubit, "X180_ge",
                                          uid=uid),
            )
        elif self.transition_to_calib == "fh":
            section.play(
                signal=self.signal_name("drive", qubit),
                pulse=qt_ops.quantum_gate(qubit, "X180_ge",
                                          uid=f"{qubit.uid}_prep_e"),
            )
            uid = f"{qubit.uid}_prep_f"
            section.play(
                signal=self.signal_name("drive_ef", qubit),
                pulse=qt_ops.quantum_gate(qubit, "X180_ef",
                                          uid=uid),
            )
        else:
            raise ValueError(
                f"Transitions name {self.transition_to_calib} "
                f"not recognised. Please used one of "
                f'["ge", "ef", "fh"].'
            )

    def analyse_experiment(self):
        self.new_qubit_parameters = {}
        self.fit_results = {}
        ts = self.timestamp if self.timestamp is not None else ''
        for qubit in self.qubits:
            # extract data
            handle = f"{self.experiment_name}_{qubit.uid}"
            do_pca = self.analysis_metainfo.get("do_pca", False)
            data_dict = ana_hlp.extract_and_rotate_data_1d(
                self.results, handle, cal_states=self.cal_states, do_pca=do_pca)
            num_cal_traces = data_dict["num_cal_traces"]

            # configure plot: data is plotted in analyse_experiment_qubit
            fig, ax = plt.subplots()
            ax.set_xlabel(self.results.get_axis_name(handle)[0])
            ax.set_ylabel("Principal Component (a.u)" if num_cal_traces == 0 else
                          f"$|{self.cal_states[-1]}\\rangle$-State Population")
            ax.set_title(f'{ts}_{handle}')
            # run the analysis from the children
            self.analyse_experiment_qubit(qubit, data_dict, fig, ax)
            if self.save:
                # Save the figure
                self.save_figure(fig, qubit)
                # Save fit results
                self.save_fit_results()
            if self.analysis_metainfo.get("show_figures", False):
                plt.show()
            plt.close(fig)

    def analyse_experiment_qubit(self, qubit, data_dict, figure, ax):
        """
        Method to be overriden by children.

        Args:
            qubit: qubit-class instance
            data_dict: the return dict of ana_hlp.extract_and_rotate_data_1d
            figure: figure instance
            ax: axis instance

        """
        pass


class AmplitudeRabi(SingleQubitGateTuneup):
    fallback_experiment_name = "Rabi"

    def define_experiment(self):
        self.experiment.sections = []
        # define Rabi experiment pulse sequence
        # outer loop - real-time, cyclic averaging
        self.create_acquire_rt_loop()
        self.experiment.add(self.acquire_loop)
        for i, qubit in enumerate(self.qubits):
            # create sweep
            sweep = Sweep(uid=f"{qubit.uid}_{self.experiment_name}_sweep",
                          parameters=[self.sweep_parameters_dict[qubit.uid][0]])

            # create preparation pulses section
            preparation_section = Section(
                uid=f"{qubit.uid}_preparation",
                alignment=SectionAlignment.RIGHT,
                on_system_grid=True,
            )
            # preparation pulses: ge if calibrating ef
            self.add_preparation_pulses_to_section(
                preparation_section, qubit)

            # create pulses section
            excitation_section = Section(
                uid=f"{qubit.uid}_excitation",
                alignment=SectionAlignment.LEFT,
                on_system_grid=True,
                play_after=f"{qubit.uid}_preparation",
            )
            # pulse to calibrate
            drive_pulse = qt_ops.quantum_gate(
                qubit, f'X180_{self.transition_to_calib}')
            # amplitude is scaled w.r.t this value
            drive_pulse.amplitude = 1
            excitation_section.play(
                signal=self.signal_name(
                    f"drive{self.drive_signal_suffix}", qubit),
                pulse=drive_pulse,
                amplitude=self.sweep_parameters_dict[qubit.uid][0]
            )

            # excitation_section.delay(signal=f"drive_{qubit.uid}", time=10e-9)
            # create readout + acquire sections
            measure_sections = self.create_measure_acquire_sections(
                uid=f"{qubit.uid}_readout",
                qubit=qubit,
                play_after=f"{qubit.uid}_excitation",
            )

            # add sweep and sections to acquire loop rt
            self.acquire_loop.add(sweep)
            sweep.add(preparation_section)
            sweep.add(excitation_section)
            sweep.add(measure_sections)
            self.add_cal_states_sections(qubit)

    def analyse_experiment_qubit(self, qubit, data_dict, figure, ax):
        # plot data
        ax.plot(data_dict["sweep_points_w_cal_tr"],
                data_dict["data_rotated_w_cal_tr"], 'o', zorder=2)
        if self.analysis_metainfo.get('do_fitting', True):
            swpts_to_fit = data_dict["sweep_points"]
            data_to_fit = data_dict["data_rotated"]
            # fit data
            freqs_guess, phase_guess = ana_hlp.find_oscillation_frequency_and_phase(
                data_to_fit, swpts_to_fit)
            param_hints = self.analysis_metainfo.get(
                'param_hints', {
                    'frequency': {'value': 2 * np.pi * freqs_guess,
                                  'min': 0},
                    'phase': {'value': phase_guess},
                    'amplitude': {'value': abs(max(data_to_fit) -
                                               min(data_to_fit)) / 2,
                                  'min': 0},
                    'offset': {'value': np.mean(data_to_fit)}
                })
            fit_res = ana_hlp.fit_data_lmfit(
                fit_mods.oscillatory, swpts_to_fit, data_to_fit,
                param_hints=param_hints)
            self.fit_results[qubit.uid] = fit_res

            freq_fit = unc.ufloat(fit_res.params['frequency'].value,
                                  fit_res.params['frequency'].stderr)
            phase_fit = unc.ufloat(fit_res.params['phase'].value,
                                  fit_res.params['phase'].stderr)
            pi_amps_top, pi_amps_bottom, pi2_amps_rise, pi2_amps_fall = \
                ana_hlp.get_pi_pi2_xvalues_on_cos(
                    swpts_to_fit, freq_fit, phase_fit)
            # if pca is done, it can happen that the pi-pulse amplitude
            # is in pi_amps_bottom and the pi/2-pulse amplitude in pi2_amps_fall
            pi_amps = np.sort(np.concatenate([pi_amps_top, pi_amps_bottom]))
            pi2_amps = np.sort(np.concatenate([pi2_amps_rise, pi2_amps_fall]))
            pi2_amp = pi2_amps[0]
            pi_amp = pi_amps[pi_amps > pi2_amp][0]
            self.new_qubit_parameters[qubit.uid] = {
                'amplitude_pi': pi_amp.nominal_value,
                'amplitude_pi2': pi2_amp.nominal_value,
                'pi_amps': [pia.nominal_value for pia in pi_amps],
                'pi2_amps': [pi2a.nominal_value for pi2a in pi_amps]
            }

            # plot fit
            swpts_fine = np.linspace(swpts_to_fit[0], swpts_to_fit[-1], 501)
            ax.plot(swpts_fine, fit_res.model.func(
                swpts_fine, **fit_res.best_values), 'r-', zorder=1)
            plt.plot(pi_amp.nominal_value, fit_res.model.func(
                pi_amp.nominal_value, **fit_res.best_values), 'sk', zorder=3,
                     markersize=plt.rcParams['lines.markersize'] + 1)
            plt.plot(pi2_amp.nominal_value, fit_res.model.func(
                pi2_amp.nominal_value, **fit_res.best_values), 'sk', zorder=3,
                     markersize=plt.rcParams['lines.markersize'] + 1)
            # textbox
            old_pi_amp = qubit.parameters.drive_parameters_ef["amplitude_pi"] if \
                'f' in self.transition_to_calib else \
                qubit.parameters.drive_parameters_ge["amplitude_pi"]
            old_pi2_amp = qubit.parameters.drive_parameters_ef["amplitude_pi2"] if \
                'f' in self.transition_to_calib else \
                qubit.parameters.drive_parameters_ge["amplitude_pi2"]
            textstr = '$A_{\\pi}$: ' + \
                      f'{pi_amp.nominal_value:.4f} $\\pm$ {pi_amp.std_dev:.4f}'
            textstr += '\nCurrent $A_{\\pi}$: ' + f'{old_pi_amp:.4f}'
            ax.text(0, -0.15, textstr, ha='left', va='top',
                    transform=ax.transAxes)
            textstr = '$A_{\\pi/2}$: ' + \
                      f'{pi2_amp.nominal_value:.4f} $\\pm$ {pi2_amp.std_dev:.4f}'
            textstr += '\nCurrent $A_{\\pi/2}$: ' + f'{old_pi2_amp:.4f}'
            ax.text(0.69, -0.15, textstr, ha='left', va='top',
                    transform=ax.transAxes)

    def update_qubit_parameters(self):
        for qubit in self.qubits:
            dr_pars = qubit.parameters.drive_parameters_ef if \
                'f' in self.transition_to_calib else \
                qubit.parameters.drive_parameters_ge
            dr_pars['amplitude_pi'] = \
                self.new_qubit_parameters[qubit.uid]['amplitude_pi']
            dr_pars['amplitude_pi2'] = \
                self.new_qubit_parameters[qubit.uid]['amplitude_pi2']


class Ramsey(SingleQubitGateTuneup):
    fallback_experiment_name = "Ramsey"

    def define_experiment(self):
        self.experiment.sections = []
        self.create_acquire_rt_loop()
        self.experiment.add(self.acquire_loop)
        # from the delays sweep parameters, create sweep parameters for
        # half the total delay time and for the phase of the second X90 pulse
        detuning = self.experiment_metainfo.get('detuning')
        if detuning is None:
            raise ValueError("Please provide detuning in experiment_metainfo.")
        swp_pars_phases = []
        for qubit in self.qubits:
            delays = deepcopy(self.sweep_parameters_dict[qubit.uid][0].values)
            pl = qubit.parameters.drive_parameters_ef["length"] \
                if 'f' in self.transition_to_calib else \
                qubit.parameters.drive_parameters_ge["length"]
            swp_pars_phases += [
                SweepParameter(
                    uid=f"x90_phases_{qubit.uid}",
                    values=((delays - delays[0] + pl) *
                            detuning[qubit.uid] * 2 * np.pi) % (2 * np.pi)
                )
            ]
        swp_pars_delays = [self.sweep_parameters_dict[qubit.uid][0]
                           for qubit in self.qubits]

        # create joint sweep for all qubits
        sweep = Sweep(
            uid=f"{self.experiment_name}_sweep",
            parameters=swp_pars_delays + swp_pars_phases)
        self.acquire_loop.add(sweep)
        for i, qubit in enumerate(self.qubits):
            # create preparation pulses section
            preparation_section = Section(
                uid=f"{qubit.uid}_preparation",
                alignment=SectionAlignment.RIGHT,
                on_system_grid=True,
            )
            # preparation pulses: ge if calibrating ef
            self.add_preparation_pulses_to_section(preparation_section, qubit)
            # Ramsey pulses
            ramsey_drive_pulse = qt_ops.quantum_gate(
                qubit, f"X90_{self.transition_to_calib}"
            )

            # create ramsey-pulses section
            excitation_section = Section(
                uid=f"{qubit.uid}_excitation",
                alignment=SectionAlignment.RIGHT,
                on_system_grid=True,
                play_after=f"{qubit.uid}_preparation",
            )
            excitation_section.play(
                signal=self.signal_name(
                    f"drive{self.drive_signal_suffix}", qubit),
                pulse=ramsey_drive_pulse
            )
            excitation_section.delay(
                signal=self.signal_name(
                    f'drive{self.drive_signal_suffix}', qubit),
                time=swp_pars_delays[i])
            excitation_section.play(
                signal=self.signal_name(
                    f"drive{self.drive_signal_suffix}", qubit),
                pulse=ramsey_drive_pulse,
                phase=swp_pars_phases[i]
            )

            # create readout + acquire sections
            measure_sections = self.create_measure_acquire_sections(
                uid=f"{qubit.uid}_readout",
                qubit=qubit,
                play_after=f"{qubit.uid}_excitation",
            )

            # add sweep and sections to acquire loop rt
            sweep.add(preparation_section)
            sweep.add(excitation_section)
            sweep.add(measure_sections)
            self.add_cal_states_sections(qubit)

    def analyse_experiment_qubit(self, qubit, data_dict, figure, ax):
        delays_offset = qubit.parameters.drive_parameters_ef["length"] \
            if 'f' in self.transition_to_calib else \
            qubit.parameters.drive_parameters_ge["length"]
        # plot data with correct scaling
        ax.plot((data_dict["sweep_points_w_cal_tr"] + delays_offset) * 1e6,
                data_dict["data_rotated_w_cal_tr"], 'o', zorder=2)
        ax.set_xlabel("Pulse Separation, $\\tau$ ($\\mu$s)")
        if self.analysis_metainfo.get('do_fitting', True):
            swpts_to_fit = data_dict["sweep_points"] + delays_offset
            data_to_fit = data_dict["data_rotated"]
            # fit data
            freqs_guess, phase_guess = ana_hlp.find_oscillation_frequency_and_phase(
                data_to_fit, swpts_to_fit)
            param_hints = self.analysis_metainfo.get(
                'param_hints', {
                    'frequency': {'value': freqs_guess},
                    'phase': {'value': phase_guess},
                    'decay_time': {'value': 2 / 3 * max(swpts_to_fit),
                                   'min': 0},
                    'amplitude': {'value': 0.5,
                                  'vary': False},
                    'oscillation_offset': {'value': 0,
                                           'vary': 'f' in self.cal_states},
                    'exponential_offset': {'value': np.mean(data_to_fit)},
                    'decay_exponent': {'value': 1, 'vary': False},
                })
            fit_res = ana_hlp.fit_data_lmfit(
                fit_mods.oscillatory_decay_new, swpts_to_fit, data_to_fit,
                param_hints=param_hints)
            self.fit_results[qubit.uid] = fit_res

            t2_star = fit_res.best_values['decay_time']
            t2_star_err = fit_res.params['decay_time'].stderr
            freq_fit = fit_res.best_values['frequency']
            freq_fit_err = fit_res.params['frequency'].stderr
            old_qb_freq = qubit.parameters.resonance_frequency_ef \
                if 'f' in self.transition_to_calib else \
                qubit.parameters.resonance_frequency_ge
            introduced_detuning = self.experiment_metainfo["detuning"][qubit.uid]
            print(old_qb_freq, introduced_detuning, freq_fit)
            # new_qb_freq = old_qb_freq - (introduced_detuning - freq_fit)
            new_qb_freq = old_qb_freq + introduced_detuning - freq_fit
            self.new_qubit_parameters[qubit.uid] = {
                'resonance_frequency': new_qb_freq,
                'T2_star': t2_star
            }

            # plot fit
            swpts_fine = np.linspace(swpts_to_fit[0], swpts_to_fit[-1], 501)
            ax.plot(swpts_fine * 1e6, fit_res.model.func(
                swpts_fine, **fit_res.best_values), 'r-', zorder=1)
            textstr = (f'New qubit frequency: {new_qb_freq / 1e9:.6f} GHz '
                       f'$\\pm$ {freq_fit_err / 1e6:.4f} MHz')
            textstr += f'\nOld qubit frequency: {old_qb_freq / 1e9:.6f} GHz'
            textstr += (f'\nDiff new-old qubit frequency: '
                        f'{(new_qb_freq - old_qb_freq) / 1e6:.6f} MHz')
            textstr += f'\nIntroduced detuning: {introduced_detuning / 1e6:.2f} MHz'
            textstr += (f'\nFitted frequency: {freq_fit / 1e6:.6f} '
                        f'$\\pm$ {freq_fit_err / 1e6:.4f} MHz')
            textstr += (f'\n$T_2^*$: {t2_star * 1e6:.4f} $\\pm$ '
                        f'{t2_star_err * 1e6:.4f} $\\mu$s')
            ax.text(0, -0.15, textstr, ha='left', va='top',
                    transform=ax.transAxes)

    def update_qubit_parameters(self):
        for qubit in self.qubits:
            new_freq = self.new_qubit_parameters[qubit.uid]['resonance_frequency']
            if 'f' in self.transition_to_calib:
                qubit.parameters.resonance_frequency_ef = new_freq
            else:
                qubit.parameters.resonance_frequency_ge = new_freq


class QScale(SingleQubitGateTuneup):
    fallback_experiment_name = "QScale"

    def define_experiment(self):
        self.experiment.sections = []
        self.create_acquire_rt_loop()
        self.experiment.add(self.acquire_loop)
        for i, qubit in enumerate(self.qubits):
            tn = self.transition_to_calib
            X90_pulse = qt_ops.quantum_gate(qubit, f"X90_{tn}")
            X180_pulse = qt_ops.quantum_gate(qubit, f"X180_{tn}")
            Y180_pulse = qt_ops.quantum_gate(qubit, f"Y180_{tn}")
            Ym180_pulse = qt_ops.quantum_gate(qubit, f"mY180_{tn}")
            pulse_ids = ["xy", "xx", "xmy"]
            pulses_2nd = [Y180_pulse, X180_pulse, Ym180_pulse]

            swp = self.sweep_parameters_dict[qubit.uid][0]
            # create sweep
            sweep = Sweep(
                uid=f"{qubit.uid}_{self.experiment_name}_sweep", parameters=[swp]
            )
            self.acquire_loop.add(sweep)
            # create pulses sections
            for i, pulse_2nd in enumerate(pulses_2nd):
                id = pulse_ids[i]
                play_after = (
                    f"{qubit.uid}_{pulse_ids[i - 1]}_section_meas" if i > 0 else None
                )

                excitation_section = Section(
                    uid=f"{qubit.uid}_{id}_section",
                    play_after=play_after,
                    alignment=SectionAlignment.RIGHT,
                )
                # preparation pulses: ge if calibrating ef
                self.add_preparation_pulses_to_section(excitation_section, qubit)
                # qscale pulses
                excitation_section.play(
                    signal=self.signal_name("drive", qubit),
                    pulse=X90_pulse,
                    pulse_parameters={"beta": swp},
                )
                excitation_section.play(
                    signal=self.signal_name("drive", qubit),
                    pulse=pulse_2nd,
                    pulse_parameters={"beta": swp},
                )

                # create readout + acquire sections
                measure_sections = self.create_measure_acquire_sections(
                    uid=f"{qubit.uid}_{id}_section_meas",
                    qubit=qubit,
                    play_after=f"{qubit.uid}_{id}_section",
                )

                # Add sections to sweep
                sweep.add(excitation_section)
                sweep.add(measure_sections)

            self.add_cal_states_sections(qubit)


class T1(SingleQubitGateTuneup):
    fallback_experiment_name = "T1"

    def define_experiment(self):
        self.experiment.sections = []
        self.create_acquire_rt_loop()
        self.experiment.add(self.acquire_loop)

        # create joint sweep for all qubits
        sweep = Sweep(
            uid=f"{self.experiment_name}_sweep",
            parameters=[self.sweep_parameters_dict[qubit.uid][0]
                        for qubit in self.qubits]
        )
        self.acquire_loop.add(sweep)
        for i, qubit in enumerate(self.qubits):

            # create preparation-pulses section
            preparation_section = Section(
                uid=f"{qubit.uid}_preparation",
                alignment=SectionAlignment.RIGHT,
                on_system_grid=True,
            )
            # preparation pulses: ge if calibrating ef
            self.add_preparation_pulses_to_section(preparation_section, qubit)

            # create excitation-pulses section
            excitation_section = Section(
                uid=f"{qubit.uid}_excitation",
                alignment=SectionAlignment.RIGHT,
                on_system_grid=True,
                play_after=f"{qubit.uid}_preparation"
            )
            # add x180 pulse
            x180_pulse = qt_ops.quantum_gate(qubit, f"X180_{self.transition_to_calib}")
            excitation_section.play(
                signal=self.signal_name(
                    f"drive{self.drive_signal_suffix}", qubit),
                pulse=x180_pulse,
            )
            # add delay
            excitation_section.delay(
                signal=self.signal_name(
                    f"drive{self.drive_signal_suffix}", qubit),
                time=self.sweep_parameters_dict[qubit.uid][0]
            )

            # create readout + acquire sections
            measure_sections = self.create_measure_acquire_sections(
                uid=f"{qubit.uid}_readout",
                qubit=qubit,
                play_after=f"{qubit.uid}_excitation",
            )

            # add sweep and sections to acquire loop rt
            sweep.add(preparation_section)
            sweep.add(excitation_section)
            sweep.add(measure_sections)
            self.add_cal_states_sections(qubit)

    def analyse_experiment_qubit(self, qubit, data_dict, figure, ax):
        # plot data with correct scaling
        ax.plot(data_dict["sweep_points_w_cal_tr"] * 1e6,
                data_dict["data_rotated_w_cal_tr"], 'o', zorder=2)
        ax.set_xlabel("Pulse Delay, $\\tau$ ($\\mu$s)")
        if self.analysis_metainfo.get('do_fitting', True):
            swpts_to_fit = data_dict["sweep_points"]
            data_to_fit = data_dict["data_rotated"]
            # fit data
            param_hints = self.analysis_metainfo.get(
                'param_hints', {
                    'decay_rate': {'value': 3 * max(swpts_to_fit) / 2},
                    'amplitude': {'value': abs(max(data_to_fit) -
                                               min(data_to_fit)) / 2,
                                  'min': 0},
                    'offset': {'value': 0, 'vary': False}
                })
            fit_res = ana_hlp.fit_data_lmfit(
                fit_mods.exponential_decay, swpts_to_fit, data_to_fit,
                param_hints=param_hints)
            self.fit_results[qubit.uid] = fit_res

            dec_rt = unc.ufloat(fit_res.params['decay_rate'].value,
                                fit_res.params['decay_rate'].stderr)
            t1 = 1 / dec_rt
            self.new_qubit_parameters[qubit.uid] = {'T1': t1.nominal_value}

            # plot fit
            swpts_fine = np.linspace(swpts_to_fit[0], swpts_to_fit[-1], 501)
            ax.plot(swpts_fine * 1e6, fit_res.model.func(
                swpts_fine, **fit_res.best_values), 'r-', zorder=1)
            textstr = (f'$T_1$: {t1.nominal_value * 1e6:.4f} $\\pm$ '
                       f'{t1.std_dev * 1e6:.4f} $\\mu$s')
            ax.text(0, -0.15, textstr, ha='left', va='top',
                    transform=ax.transAxes)


class Echo(SingleQubitGateTuneup):
    fallback_experiment_name = "Echo"

    def define_experiment(self):
        self.experiment.sections = []
        self.create_acquire_rt_loop()
        self.experiment.add(self.acquire_loop)
        # from the delays sweep parameters, create sweep parameters for
        # half the total delay time and for the phase of the second X90 pulse
        detuning = self.experiment_metainfo.get('detuning')
        if detuning is None:
            raise ValueError("Please provide detuning in experiment_metainfo.")
        swp_pars_half_delays = []
        swp_pars_phases = []
        for qubit in self.qubits:
            delays = self.sweep_parameters_dict[qubit.uid][0].values
            pl = qubit.parameters.drive_parameters_ef["length"] \
                if 'f' in self.transition_to_calib else \
                qubit.parameters.drive_parameters_ge["length"]
            swp_pars_half_delays += [
                SweepParameter(
                    uid=f"echo_delays_{qubit.uid}",
                    values=0.5 * (delays - pl))  # subtract the echo-pulse length
            ]
            swp_pars_phases += [
                SweepParameter(
                    uid=f"echo_phases_{qubit.uid}",
                    values=((delays - delays[0] + pl) *
                            detuning[qubit.uid] * 2 * np.pi) % (2 * np.pi)
                )
            ]

        # create joint sweep for all qubits
        sweep = Sweep(uid=f"{self.experiment_name}_sweep",
                      parameters=swp_pars_half_delays + swp_pars_phases)
        self.acquire_loop.add(sweep)
        for i, qubit in enumerate(self.qubits):

            # create preparation pulses section
            preparation_section = Section(
                uid=f"{qubit.uid}_preparation",
                alignment=SectionAlignment.RIGHT,
                on_system_grid=True,
            )
            # preparation pulses: ge if calibrating ef
            self.add_preparation_pulses_to_section(preparation_section, qubit)

            # create excitation section
            excitation_section = Section(
                uid=f"{qubit.uid}_excitation",
                alignment=SectionAlignment.RIGHT,
                on_system_grid=True,
                play_after=f"{qubit.uid}_preparation",
            )
            # Echo pulses and delays
            ramsey_drive_pulse = qt_ops.quantum_gate(
                qubit, f"X90_{self.transition_to_calib}"
            )
            echo_drive_pulse = qt_ops.quantum_gate(
                qubit, f"X180_{self.transition_to_calib}"
            )
            excitation_section.play(
                signal=self.signal_name(
                    f"drive{self.drive_signal_suffix}", qubit),
                pulse=ramsey_drive_pulse
            )
            excitation_section.delay(
                signal=self.signal_name(
                    f"drive{self.drive_signal_suffix}", qubit),
                time=swp_pars_half_delays[i]
            )
            excitation_section.play(
                signal=self.signal_name(
                    f"drive{self.drive_signal_suffix}", qubit),
                pulse=echo_drive_pulse
            )
            excitation_section.delay(
                signal=self.signal_name(
                    f"drive{self.drive_signal_suffix}", qubit),
                time=swp_pars_half_delays[i]
            )
            excitation_section.play(
                signal=self.signal_name(
                    f"drive{self.drive_signal_suffix}", qubit),
                pulse=ramsey_drive_pulse,
                phase=swp_pars_phases[i]
            )

            # create readout + acquire sections
            measure_sections = self.create_measure_acquire_sections(
                uid=f"{qubit.uid}_readout",
                qubit=qubit,
                play_after=f"{qubit.uid}_excitation",
            )

            # add sweep and sections to acquire loop rt
            sweep.add(preparation_section)
            sweep.add(excitation_section)
            sweep.add(measure_sections)
            self.add_cal_states_sections(qubit)

    def analyse_experiment_qubit(self, qubit, data_dict, figure, ax):
        # plot data with correct scaling
        ax.plot(data_dict["sweep_points_w_cal_tr"] * 1e6,
                data_dict["data_rotated_w_cal_tr"], 'o', zorder=2)
        ax.set_xlabel("Pulse Separation, $\\tau$ ($\\mu$s)")

        if self.analysis_metainfo.get('do_fitting', True):
            swpts_to_fit = data_dict["sweep_points"]
            data_to_fit = data_dict["data_rotated"]
            # fit data
            freqs_guess, phase_guess = ana_hlp.find_oscillation_frequency_and_phase(
                data_to_fit, swpts_to_fit)
            param_hints = self.analysis_metainfo.get(
                'param_hints', {
                    'frequency': {'value': 2 * np.pi * freqs_guess},
                    'phase': {'value': phase_guess},
                    'decay_rate': {'value': 3 * max(swpts_to_fit) / 2,
                                   'min': 0},
                    'amplitude': {'value': 0.5,
                                  'vary': False},
                    'offset': {'value': np.mean(data_to_fit)}
                })
            fit_res = ana_hlp.fit_data_lmfit(
                fit_mods.oscillatory_decay, swpts_to_fit, data_to_fit,
                param_hints=param_hints)
            self.fit_results[qubit.uid] = fit_res

            dec_rt = unc.ufloat(fit_res.params['decay_rate'].value,
                                fit_res.params['decay_rate'].stderr)
            t2 = 1 / dec_rt
            self.new_qubit_parameters[qubit.uid] = {'T2': t2.nominal_value}

            # plot fit
            swpts_fine = np.linspace(swpts_to_fit[0], swpts_to_fit[-1], 501)
            ax.plot(swpts_fine * 1e6, fit_res.model.func(
                swpts_fine, **fit_res.best_values), 'r-', zorder=1)
            textstr = (f'$T_2$: {t2.nominal_value * 1e6:.4f} $\\pm$ '
                       f'{t2.std_dev * 1e6:.4f} $\\mu$s')
            ax.text(0, -0.15, textstr, ha='left', va='top',
                    transform=ax.transAxes)


class RamseyParking(Ramsey):
    fallback_experiment_name = "RamseyParking"

    def define_experiment(self):
        super().define_experiment()
        self.experiment.sections = []
        qubit = self.qubits[0]  # TODO: parallelize
        # voltage sweep
        nt_sweep_par = self.sweep_parameters_dict[qubit.uid][1]
        nt_sweep = Sweep(
            uid=f"neartime_voltage_sweep_{qubit.uid}",
            parameters=[nt_sweep_par],
        )
        self.experiment.add(nt_sweep)
        ntsf = self.experiment_metainfo.get("neartime_callback_function", None)
        if ntsf is None:
            raise ValueError(
                "Please provide the neartime callback function for the voltage"
                "sweep in experiment_metainfo['neartime_sweep_prameter'].")
        # all near-time callback functions have the format
        # func(session, sweep_param_value, qubit)
        nt_sweep.call(ntsf, voltage=nt_sweep_par, qubit=qubit)
        # self.create_acquire_rt_loop()
        nt_sweep.add(self.acquire_loop)

    def analyse_experiment(self):
        self.new_qubit_parameters = {}
        self.fit_results = {}
        ts = self.timestamp if self.timestamp is not None else ''
        for qubit in self.qubits:
            delays_offset = qubit.parameters.drive_parameters_ef["length"] \
                if 'f' in self.transition_to_calib else \
                qubit.parameters.drive_parameters_ge["length"]
            # extract data
            handle = f"{self.experiment_name}_{qubit.uid}"
            do_pca = self.analysis_metainfo.get("do_pca", False)
            data_dict = ana_hlp.extract_and_rotate_data_2d(
                self.results, handle, cal_states=self.cal_states, do_pca=do_pca)
            num_cal_traces = data_dict["num_cal_traces"]

            if self.analysis_metainfo.get("do_fitting", True):
                voltages = data_dict["sweep_points_nt"]
                all_fit_results = {}
                all_new_qb_pars = {}
                # run Ramsey analysis
                data_to_fit_2d = data_dict["data_rotated"]
                data_rotated_w_cal_tr_2d = data_dict["data_rotated_w_cal_tr"]
                for i in range(data_to_fit_2d.shape[0]):
                    data_to_fit = data_to_fit_2d[i, :]
                    data_dict_tmp = deepcopy(data_dict)
                    data_dict_tmp["data_rotated"] = data_to_fit
                    data_dict_tmp["data_rotated_w_cal_tr"] = data_rotated_w_cal_tr_2d[i, :]

                    fig, ax = plt.subplots()
                    ax.set_xlabel(self.results.get_axis_name(handle)[1])
                    ax.set_ylabel("Principal Component (a.u)" if
                                  (num_cal_traces == 0 or do_pca) else
                                  f"$|{self.cal_states[-1]}\\rangle$-State Population")
                    ax.set_title(f'{ts}_{handle}')
                    # run ramsey analysis
                    self.analyse_experiment_qubit(qubit, data_dict_tmp, fig, ax)
                    if self.save:
                        # Save the figure
                        fig_name = (f"{self.timestamp}_Ramsey"
                                    f"_{qubit.uid}_{voltages[i]:.3f}V")
                        self.save_figure(fig, qubit, fig_name)
                    plt.close(fig)
                    all_fit_results[i] = self.fit_results[qubit.uid]
                    all_new_qb_pars[i] = self.new_qubit_parameters[qubit.uid]

                self.new_qubit_parameters[qubit.uid] = all_new_qb_pars
                self.fit_results[qubit.uid] = all_fit_results
                # fit qubit frequencies vs voltage
                qubit_frequencies = np.array([
                    self.new_qubit_parameters[qubit.uid][i]['resonance_frequency']
                    for i in range(len(voltages))])

                # figure out whether voltages vs freqs is convex or concave
                take_extremum_fit, scf = (np.argmax, 1) if (
                    ana_hlp.is_data_convex(voltages, qubit_frequencies)) \
                    else (np.argmin, -1)
                # optimal parking parameters at the extremum of
                # voltages vs frequencies
                f0 = qubit_frequencies[take_extremum_fit(qubit_frequencies)]
                V0 = voltages[take_extremum_fit(qubit_frequencies)]
                param_hints = {
                    'V0': {'value': V0},
                    'f0': {'value': f0},
                    'fv': {'value': scf * (max(qubit_frequencies) -
                                           min(qubit_frequencies))},
                }
                fit_res = ana_hlp.fit_data_lmfit(
                    fit_mods.transmon_voltage_dependence_quadratic,
                    voltages, qubit_frequencies, param_hints=param_hints)
                f0, f0err = fit_res.best_values['f0'], fit_res.params['f0'].stderr
                V0, V0err = fit_res.best_values['V0'], fit_res.params['V0'].stderr
                self.fit_results[qubit.uid]["parking"] = fit_res
                self.new_qubit_parameters[qubit.uid]["parking"] = {
                    "resonance_frequency": f0,
                    "dc_voltage_parking": V0
                }
                # plot data + fit
                fig, ax = plt.subplots()
                ax.set_xlabel(self.results.get_axis_name(handle)[0])
                ax.set_ylabel("Qubit Frequency, $f_{qb}$ (GHz)")
                ax.set_title(f'{ts}_{handle}')
                ax.plot(voltages, qubit_frequencies / 1e9, 'o', zorder=2)
                # plot fit
                voltages_fine = np.linspace(voltages[0], voltages[-1], 501)
                ax.plot(voltages_fine, fit_res.model.func(
                    voltages_fine, **fit_res.best_values) / 1e9, 'r-')
                if voltages[0] <= V0 <= voltages[-1]:
                    ax.plot(V0, f0 / 1e9, 'sk',
                            markersize=plt.rcParams['lines.markersize'] + 1)
                textstr = f"Parking voltage: {V0:.4f} $\\pm$ {V0err:.4f} V"
                textstr += f"\nParking frequency: {f0 / 1e9:.4f} $\\pm$ {f0err / 1e9:.4f} GHz"
                ax.text(0, -0.15, textstr, ha='left', va='top',
                        transform=ax.transAxes)

                # save figures and results
                if self.save:
                    # Save the figure
                    self.save_figure(fig, qubit)
                    if len(self.fit_results) > 0:
                        # Save fit results
                        self.save_fit_results()
                if self.analysis_metainfo.get("show_figures", False):
                    plt.show()
                plt.close(fig)

    def update_qubit_parameters(self):
        for qubit in self.qubits:
            new_qb_pars = self.new_qubit_parameters[qubit.uid]
            qubit.parameters.resonance_frequency_ge = new_qb_pars["parking"][
                "resonance_frequency"]
            if "dc_voltage_parking" in new_qb_pars:
                qubit.parameters.user_defined["dc_voltage_parking"] = \
                    new_qb_pars["parking"]["dc_voltage_parking"]