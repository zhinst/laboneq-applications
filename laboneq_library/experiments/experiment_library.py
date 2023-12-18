import json
import os
import sys
import time
import pickle
from copy import deepcopy
from logging import StreamHandler

from ruamel.yaml import YAML

ryaml = YAML()

import traceback
import logging

from . import quantum_operations as qt_ops
from laboneq.dsl.experiment.builtins import *  # noqa: F403
from laboneq.simple import *  # noqa: F403
from laboneq_library import calibration_helpers as calib_hlp
from laboneq_library.analysis import analysis_helpers as ana_hlp

log = logging.getLogger(__name__)
log.addHandler(StreamHandler(stream=sys.stderr))
log.setLevel(logging.WARNING)


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
                    reset_delay=qubit.parameters.reset_delay_length,
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
                    reset_delay=qubit.parameters.reset_delay_length,
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
                            integration_kernel=qt_ops.integration_kernel(qubit),
                            reset_delay=qubit.parameters.reset_delay_length,
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
                            reset_delay=1e-6,
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
                            reset_delay=qubit.parameters.reset_delay_length,
                        )

    return exp_ramsey()


###### Class - based  ######
##### Added by Steph   #####


class StatePreparationMixin:
    def create_preparation(
        self,
        qubit,
        state_to_prepare="e",
        section_uid_suffix="",
        play_after_sections=None,
        add_measure_acquire_sections=False,
        acquire_handle_name_suffix="",
    ):
        if len(section_uid_suffix) > 0:
            section_uid_suffix = f"_{section_uid_suffix}"
        if state_to_prepare == "g":
            preparation_sections = []
            if add_measure_acquire_sections:
                g_measure_section = self.create_measure_acquire_sections(
                    qubit=qubit,
                    handle_suffix=f"{acquire_handle_name_suffix}_g",
                )
                preparation_sections = [g_measure_section]
        elif state_to_prepare == "e":
            e_section = Section(
                uid=f"{qubit.uid}_prep_e{section_uid_suffix}",
                play_after=play_after_sections,
                on_system_grid=True,
            )
            e_section.play(
                signal=self.signal_name("drive", qubit),
                pulse=qt_ops.quantum_gate(qubit, "X180_ge", uid=f"{qubit.uid}_prep_e"),
            )
            preparation_sections = [e_section]
            if add_measure_acquire_sections:
                e_measure_section = self.create_measure_acquire_sections(
                    qubit=qubit,
                    play_after=f"{qubit.uid}_prep_e{section_uid_suffix}",
                    handle_suffix=f"{acquire_handle_name_suffix}_e",
                )
                preparation_sections += [e_measure_section]
        elif state_to_prepare == "f":
            # prepare e state
            e_section = Section(
                uid=f"{qubit.uid}_prep_f_pulse_e{section_uid_suffix}",
                play_after=play_after_sections,
                on_system_grid=True,
            )
            e_section.play(
                signal=self.signal_name("drive", qubit),
                pulse=qt_ops.quantum_gate(
                    qubit, "X180_ge", uid=f"{qubit.uid}_prep_f_pulse_e"
                ),
            )
            # prepare f state
            f_section = Section(
                uid=f"{qubit.uid}_prep_f_pulse_f{section_uid_suffix}",
                play_after=play_after_sections + [e_section],
                on_system_grid=True,
            )
            f_section.play(
                signal=self.signal_name("drive_ef", qubit),
                pulse=qt_ops.quantum_gate(
                    qubit, "X180_ef", uid=f"{qubit.uid}_prep_f_pulse_f"
                ),
            )
            preparation_sections = [e_section, f_section]
            if add_measure_acquire_sections:
                f_measure_section = self.create_measure_acquire_sections(
                    qubit=qubit,
                    play_after=f"{qubit.uid}_prep_f_pulse_f{section_uid_suffix}",
                    handle_suffix=f"{acquire_handle_name_suffix}_f",
                )
                preparation_sections += [f_measure_section]
        else:
            raise NotImplementedError(
                "Currently, only state g, e and f " "can be prepared."
            )
        return preparation_sections


class ExperimentTemplate(StatePreparationMixin):
    fallback_experiment_name = "Experiment"
    save_directory = None
    timestamp = None
    compiled_experiment = None
    results = None
    analysis_results = None
    valid_user_parameters = dict(
        experiment_metainfo=[],
        analysis_metainfo=[
            "figure_name",
            "overwrite_figures",
            "figure_formats",
        ],
    )

    def __init__(
        self,
        qubits,
        session,
        measurement_setup,
        experiment_name=None,
        signals=None,
        sweep_parameters_dict=None,
        experiment_metainfo=None,
        acquisition_metainfo=None,
        qubit_temporary_values=None,
        do_analysis=True,
        analysis_metainfo=None,
        save=True,
        data_directory=None,
        update=False,
        run=False,
        check_valid_user_parameters=True,
        **kwargs,
    ):
        self.qubits = qubits
        self.session = session
        self.measurement_setup = measurement_setup

        self.sweep_parameters_dict = deepcopy(sweep_parameters_dict)
        if self.sweep_parameters_dict is None:
            self.sweep_parameters_dict = {}
        for key, sd in self.sweep_parameters_dict.items():
            if not hasattr(sd, "__iter__"):
                self.sweep_parameters_dict[key] = [sd]

        self.experiment_metainfo = experiment_metainfo
        if self.experiment_metainfo is None:
            self.experiment_metainfo = {}
        self.cal_states = self.experiment_metainfo.get("cal_states", None)
        if acquisition_metainfo is None:
            acquisition_metainfo = {}
        self.acquisition_metainfo = dict(count=2**12)
        # overwrite default with user-provided options
        self.acquisition_metainfo.update(acquisition_metainfo)
        if qubit_temporary_values is None:
            qubit_temporary_values = {}
        self.qubit_temporary_values = qubit_temporary_values

        self.data_directory = data_directory
        self.do_analysis = do_analysis
        self.analysis_metainfo = analysis_metainfo
        if self.analysis_metainfo is None:
            self.analysis_metainfo = {}
        self.update = update
        self.save = save
        if self.save and self.data_directory is None:
            raise ValueError(
                "save==True, but no data_directory was specified. "
                "Please provide data_directory or set save=False."
            )

        self.experiment_name = experiment_name
        if self.experiment_name is None:
            self.experiment_name = self.fallback_experiment_name
        self.create_experiment_label()
        self.generate_timestamp_save_directory()

        self.signals = signals
        if self.signals is None:
            self.signals = ["drive", "measure", "acquire"]
        (
            self.experiment_signals,
            self.experiment_signal_uids_qubit_map,
        ) = self.create_experiment_signals(self.qubits, self.signals)

        if check_valid_user_parameters:
            self.check_user_parameters_validity()
        self.create_experiment()

        self.run = run
        if self.run:
            self.autorun()

    def check_user_parameters_validity(self):
        for par in self.experiment_metainfo:
            if par not in self.valid_user_parameters["experiment_metainfo"]:
                log.warning(
                    f"Parameter '{par}' passed to experiment_metainfo "
                    f"is not recognised and will probably not have an effect. "
                    f"The valid parameters are "
                    f"{self.valid_user_parameters['experiment_metainfo']}"
                )
        for par in self.analysis_metainfo:
            if par not in self.valid_user_parameters["analysis_metainfo"]:
                log.warning(
                    f"Parameter '{par}' passed to analysis_metainfo "
                    f"is not recognised and will probably not have an effect. "
                    f"The valid parameters are "
                    f"{self.valid_user_parameters['analysis_metainfo']}"
                )

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
        """
        Define the experiment acquire loops, sweeps, sections, pulses

        To be overridden by children

        """
        pass

    def configure_experiment(self):
        """
        Set the measurement_setup calibration from the qubits calibrations.

        To be overridden by children for setting the experiment calibration.

        """
        self.update_measurement_setup()

    def create_unique_uids(self):
        from laboneq.dsl.experiment.play_pulse import PlayPulse

        uids = set()

        def rename_uid_sweep_section(sec, suffix):
            if isinstance(sec, (Sweep, Section)):
                if sec.uid in uids:
                    sec_type = "Sweep" if isinstance(sec, type(Sweep)) else "Section"
                    print(f"Renaming {sec_type} uid {sec.uid} to {sec.uid}_{suffix}")
                    sec.uid += f"_{suffix}"
                    suffix += 1
                uids.add(sec.uid)
                for ch in sec.children:
                    suffix = rename_uid_sweep_section(ch, suffix=suffix)
            elif isinstance(sec, PlayPulse):
                if sec.pulse.uid in uids:
                    print(
                        f"Renaming Pulse with uid {sec.pulse.uid} to {sec.pulse.uid}_{suffix}"
                    )
                    sec.pulse.uid += f"_{suffix}"
                    suffix += 1
                uids.add(sec.pulse.uid)
            return suffix

        for i, sec in enumerate(self.experiment.sections):
            rename_uid_sweep_section(sec, suffix=0)

    def compile_experiment(self):
        self.create_unique_uids()
        self.compiled_experiment = self.session.compile(self.experiment)

    def run_experiment(self):
        self.results = self.session.run(self.compiled_experiment)

    def analyse_experiment(self):
        # to be overridden by children
        self.analysis_results = {
            qubit.uid: dict(new_parameter_values=dict(), fit_results=None)
            for qubit in self.qubits
        }

    def update_measurement_setup(self):
        calib_hlp.update_measurement_setup_from_qubits(
            self.qubits, self.measurement_setup
        )

    def update_qubit_parameters(self):
        pass

    def update_entire_setup(self):
        self.update_qubit_parameters()
        self.update_measurement_setup()

    def generate_timestamp_save_directory(self):
        # create experiment timestamp
        self.timestamp = str(time.strftime("%Y%m%d_%H%M%S"))
        if self.data_directory is not None:
            # create experiment save_directory
            self.save_directory = os.path.abspath(
                os.path.join(
                    self.data_directory,
                    f"{self.timestamp[:8]}",
                    f"{self.timestamp[-6:]}_{self.experiment_label}",
                )
            )

    def create_save_directory(self):
        # create the save_directory inside self.data_directory
        if self.save_directory is not None and not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)

    def save_measurement_setup(self):
        """
        Saves the measurement_setup into a json file, and creates another json
        file with the meta-information passed to this class: experiment_metainfo
        and analysis_metainfo.

        The saved measurement_setup contains the setup description before the
        execution of the experiment.

        """
        self.create_save_directory()

        # Save the measurement setup
        filename = f"measurement_setup_before_experiment.json"
        filename = f"{self.timestamp}_{filename}"
        filepath = os.path.abspath(os.path.join(self.save_directory, filename))
        if filename not in os.listdir(self.save_directory):
            # only save the setup if the file does not already exist
            self.measurement_setup.save(filepath)

    def save_experiment_metainfo(self):
        # Save the meta-information
        metainfo = {
            "sweep_parameters_dict": self.sweep_parameters_dict,
            "experiment_metainfo": self.experiment_metainfo,
            "analysis_metainfo": self.analysis_metainfo,
        }
        metainfo_file = os.path.abspath(
            os.path.join(self.save_directory, f"{self.timestamp}_meta_information.p")
        )
        with open(metainfo_file, "wb") as f:
            pickle.dump(metainfo, f)

    def save_results(self, filename_suffix=""):
        if len(filename_suffix) > 0:
            filename_suffix = f"_{filename_suffix}"
        if self.results is not None:
            self.create_save_directory()
            # Save Results
            results_file = os.path.abspath(
                os.path.join(
                    self.save_directory,
                    f"{self.timestamp}_results{filename_suffix}.json",
                )
            )
            try:
                self.results.save(results_file)
            except Exception as e:
                log.warning(f"Could not save all the results: {e}")

            # Save only the acquired results as pickle: fallback in case
            # something goes wrong with the serialisation
            filename = os.path.abspath(
                os.path.join(
                    self.save_directory,
                    f"{self.timestamp}_acquired_results{filename_suffix}.p",
                )
            )
            with open(filename, "wb") as f:
                pickle.dump(self.results.acquired_results, f)

    def save_figure(self, fig, qubit, figure_name=None):
        self.create_save_directory()
        fig_name = self.analysis_metainfo.get("figure_name", figure_name)
        if fig_name is None:
            fig_name = f"{self.timestamp}_{self.experiment_name}_{qubit.uid}"
        fig_name_final = fig_name
        if not self.analysis_metainfo.get("overwrite_figures", False):
            i = 1
            # check if filename exists in self.save_directory
            while any(
                [
                    fn.endswith(f"{fig_name_final}.png")
                    for fn in os.listdir(self.save_directory)
                ]
            ):
                fig_name_final = f"{fig_name}_{i}"
                i += 1
        fig_fmts = self.analysis_metainfo.get("figure_formats", ["png"])
        for fmt in fig_fmts:
            fig.savefig(
                self.save_directory + f"\\{fig_name_final}.{fmt}",
                bbox_inches="tight",
                dpi=600,
            )

    def save_analysis_results(self, filename_suffix=""):
        if self.analysis_results is None:
            return

        new_qb_params_exist = any(
            [
                len(self.analysis_results[qubit.uid]["new_parameter_values"]) > 0
                for qubit in self.qubits
            ]
        )
        fit_results_exist = any(
            [
                self.analysis_results[qubit.uid]["fit_results"] is not None
                for qubit in self.qubits
            ]
        )
        other_ana_res_exist = any(
            [len(self.analysis_results[qubit.uid]) > 2 for qubit in self.qubits]
        )

        if new_qb_params_exist or fit_results_exist or other_ana_res_exist:
            self.create_save_directory()

            if len(filename_suffix) > 0:
                filename_suffix = f"_{filename_suffix}"

            # Save fit results as json for easier readability
            if fit_results_exist:
                fit_results_to_save_json = dict()
                for qbuid, ana_res in self.analysis_results.items():
                    fit_results = self.analysis_results[qbuid]["fit_results"]
                    # Convert lmfit results into a dictionary that can be saved
                    # as json
                    if isinstance(fit_results, dict):
                        fit_results_to_save_json[qbuid] = {}
                        for k, fr in fit_results.items():
                            fit_results_to_save_json[qbuid][
                                k
                            ] = ana_hlp.flatten_lmfit_modelresult(fr)
                    else:
                        fit_results_to_save_json[
                            qbuid
                        ] = ana_hlp.flatten_lmfit_modelresult(fit_results)
                # Save fit results into a json file
                fit_res_file = os.path.abspath(
                    os.path.join(
                        self.save_directory,
                        f"{self.timestamp}_fit_results{filename_suffix}.json",
                    )
                )
                with open(fit_res_file, "w") as file:
                    json.dump(fit_results_to_save_json, file, indent=2)

            # ana_results_to_save = deepcopy(self.analysis_results)
            # fit_results_to_pickle = dict()
            # if fit_results_exist:
            #     for qbuid, ana_res in self.analysis_results.items():
            #         fit_results = self.analysis_results[qbuid]["fit_results"]
            #         fit_results_to_pickle[qbuid] = fit_results
            #
            #         # Convert lmfit results into a dictionary that can be saved
            #         # as json
            #         if isinstance(fit_results, dict):
            #             ana_results_to_save[qbuid]["fit_results"] = {}
            #             for k, fr in fit_results.items():
            #                 ana_results_to_save["fit_results"][qbuid][k] = \
            #                     ana_hlp.flatten_lmfit_modelresult(fr)
            #         else:
            #             ana_results_to_save[qbuid]["fit_results"] = \
            #                 ana_hlp.flatten_lmfit_modelresult(fit_results)
            #
            #     # Save fit results into a pickle file
            #     filename = os.path.abspath(os.path.join(
            #         self.save_directory,
            #         f"{self.timestamp}_fit_results{filename_suffix}.p")
            #     )
            #     with open(filename, "wb") as f:
            #         pickle.dump(fit_results_to_pickle, f)

            # Save analysis_results pickle file
            ana_res_file = os.path.abspath(
                os.path.join(
                    self.save_directory,
                    f"{self.timestamp}_analysis_results{filename_suffix}.p",
                )
            )
            with open(ana_res_file, "wb") as f:
                pickle.dump(self.analysis_results, f)

    def autorun(self):
        try:
            if self.save:
                # save the measurement setup configuration before the experiment
                # execution
                self.save_measurement_setup()
                # save the meta-information
                self.save_experiment_metainfo()
            with calib_hlp.QubitTemporaryValuesContext(*self.qubit_temporary_values):
                self.define_experiment()
                self.configure_experiment()
                self.compile_experiment()
                self.run_experiment()
                if self.do_analysis:
                    self.analyse_experiment()
            if self.update:
                self.update_entire_setup()
            if self.save:
                # Save Results object
                self.save_results()
                # Save the fit results
                self.save_analysis_results()
        except Exception:
            log.error("Unhandled error during experiment!")
            log.error(traceback.format_exc())

    def create_acquire_rt_loop(self):
        self.acquire_loop = AcquireLoopRt(**self.acquisition_metainfo)

    def create_measure_acquire_sections(
        self,
        qubit,
        uid=None,
        play_after=None,
        handle_suffix="",
        integration_kernel="default",
    ):
        handle = f"{self.experiment_name}_{qubit.uid}"
        if len(handle_suffix) > 0:
            handle += f"_{handle_suffix}"

        ro_pulse = qt_ops.readout_pulse(qubit)
        if not hasattr(self, "integration_kernel"):
            # ensure the integration_kernel is created only once to avoid
            # serialisation errors
            self.integration_kernel = qubit.get_integration_kernels()
            for int_krn in self.integration_kernel:
                if hasattr(int_krn, "uid"):
                    int_krn.uid = f"integration_kernel_{qubit.uid}"
        if integration_kernel == "default":
            integration_kernel = self.integration_kernel
        if isinstance(integration_kernel, list) and len(integration_kernel) == 1:
            integration_kernel = integration_kernel[0]
        measure_acquire_section = Section(uid=uid)
        measure_acquire_section.play_after = play_after
        measure_acquire_section.measure(
            measure_signal=self.signal_name("measure", qubit),
            measure_pulse=ro_pulse,
            handle=handle,
            acquire_signal=self.signal_name("acquire", qubit),
            integration_kernel=integration_kernel,
            integration_length=qubit.parameters.readout_integration_length,
            reset_delay=qubit.parameters.reset_delay_length,
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
                qubit=qubit,
                handle_suffix="cal_trace_g",
            )
            play_after_sections = [g_measure_section]
            cal_trace_sections += [g_measure_section]
        if "e" in self.cal_states:
            # Excited state - prep pulse + msmt
            e_prep_sections = self.create_preparation(
                qubit,
                state_to_prepare="e",
                play_after_sections=play_after_sections,
                section_uid_suffix="cal_traces",
            )
            e_measure_section = self.create_measure_acquire_sections(
                qubit=qubit,
                play_after=e_prep_sections,
                handle_suffix="cal_trace_e",
            )
            play_after_sections = (
                [s for s in play_after_sections] + e_prep_sections + [e_measure_section]
            )
            cal_trace_sections += e_prep_sections + [e_measure_section]
        if "f" in self.cal_states:
            # 2nd-excited-state - prep pulse + msmt
            f_prep_sections = self.create_preparation(
                qubit,
                state_to_prepare="f",
                play_after_sections=play_after_sections,
                section_uid_suffix="cal_traces",
            )
            f_measure_section = self.create_measure_acquire_sections(
                qubit=qubit,
                play_after=f_prep_sections,
                handle_suffix="cal_trace_f",
            )
            cal_trace_sections += f_prep_sections + [f_measure_section]
        for cal_tr_sec in cal_trace_sections:
            if section_container is None:
                self.acquire_loop.add(cal_tr_sec)
            else:
                section_container.add(cal_tr_sec)


def merge_valid_user_parameters(user_parameters_list):
    valid_user_parameters = dict(
        experiment_metainfo=[],
        analysis_metainfo=[],
    )
    for user_parameters in user_parameters_list:
        if "experiment_metainfo" in user_parameters:
            valid_user_parameters["experiment_metainfo"].extend(
                user_parameters["experiment_metainfo"]
            )
        if "analysis_metainfo" in user_parameters:
            valid_user_parameters["analysis_metainfo"].extend(
                user_parameters["analysis_metainfo"]
            )
    return valid_user_parameters
