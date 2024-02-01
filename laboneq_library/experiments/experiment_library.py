import json
import os
import sys
import time
import dill as pickle
from copy import deepcopy
from logging import StreamHandler

from ruamel.yaml import YAML

ryaml = YAML()

import traceback
import logging

from . import quantum_operations as qt_ops
from laboneq.simple import *  # noqa: F403
from laboneq_library import calibration_helpers as calib_hlp
from laboneq_library.analysis import analysis_helpers as ana_hlp

log = logging.getLogger(__name__)
log.addHandler(StreamHandler(stream=sys.stderr))
log.setLevel(logging.WARNING)



###### Class - based  ######
##### Added by Steph   #####


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


class StatePreparationMixin:
    def create_state_preparation_sections(
        self,
        qubit,
        state_to_prepare="e",
        section_uid_suffix="",
        play_after=None,
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
                    play_after=play_after,
                    handle_suffix=f"{acquire_handle_name_suffix}_g",
                )
                preparation_sections = [g_measure_section]
        elif state_to_prepare == "e":
            e_section = Section(
                uid=f"{qubit.uid}_prep_e{section_uid_suffix}",
                play_after=play_after,
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
                play_after=play_after,
                on_system_grid=True,
            )
            e_section.play(
                signal=self.signal_name("drive", qubit),
                pulse=qt_ops.quantum_gate(
                    qubit, "X180_ge", uid=f"{qubit.uid}_prep_f_pulse_e"
                ),
            )
            # prepare f state
            play_after_for_f = [e_section]
            if play_after is not None:
                play_after_for_f = play_after + play_after_for_f
            f_section = Section(
                uid=f"{qubit.uid}_prep_f_pulse_f{section_uid_suffix}",
                play_after=play_after_for_f,
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

    def add_cal_states_sections(self, qubit, add_to=None):
        if self.cal_states is None:
            return

        play_after = []
        cal_trace_sections = []
        if "g" in self.cal_states:
            # create preparation pulses sections like active reset
            g_qb_prep_sec = StatePreparationMixin.create_qubit_preparation_sections(
                self, qubit, handle_suffix="cal_traces_g"
            )
            # Ground state - just a msmt
            g_measure_section = self.create_measure_acquire_sections(
                qubit=qubit,
                handle_suffix="cal_trace_g",
            )
            play_after = [g_measure_section]
            cal_trace_sections += g_qb_prep_sec + [g_measure_section]
        if "e" in self.cal_states:
            # create preparation pulses sections like active reset
            e_qb_prep_sec = StatePreparationMixin.create_qubit_preparation_sections(
                self, qubit, handle_suffix="cal_traces_e", play_after=play_after
            )
            # Excited state - prep pulse + msmt
            e_prep_sections = self.create_state_preparation_sections(
                qubit,
                state_to_prepare="e",
                play_after=play_after + e_qb_prep_sec,  # e_qb_prep_sec might be []
                section_uid_suffix="cal_traces",
            )
            e_measure_section = self.create_measure_acquire_sections(
                qubit=qubit,
                play_after=e_prep_sections,
                handle_suffix="cal_trace_e",
            )
            play_after = [e_measure_section]
            cal_trace_sections += e_qb_prep_sec + e_prep_sections + [e_measure_section]
        if "f" in self.cal_states:
            # create preparation pulses sections like active reset
            f_qb_prep_sec = StatePreparationMixin.create_qubit_preparation_sections(
                self, qubit, handle_suffix="cal_traces_f", play_after=play_after
            )
            # 2nd-excited-state - prep pulse + msmt
            f_prep_sections = self.create_state_preparation_sections(
                qubit,
                state_to_prepare="f",
                play_after=play_after + f_qb_prep_sec,  # f_qb_prep_sec might be []
                section_uid_suffix="cal_traces",
            )
            f_measure_section = self.create_measure_acquire_sections(
                qubit=qubit,
                play_after=f_prep_sections,
                handle_suffix="cal_trace_f",
            )
            cal_trace_sections += f_qb_prep_sec + f_prep_sections + [f_measure_section]
        for cal_tr_sec in cal_trace_sections:
            if add_to is None:
                self.acquire_loop.add(cal_tr_sec)
            else:
                add_to.add(cal_tr_sec)

    def create_active_reset_sections(
        self, qubit, states_to_reset=("g", "e"), handle_suffix=None, play_after=None
    ):
        handle = "active_reset"
        if handle_suffix is not None:
            handle_suffix = f"_{handle_suffix}"
        else:
            handle_suffix = ""
        # create readout + acquire sections
        measure_acquire_section = self.create_measure_acquire_sections(
            qubit=qubit,
            play_after=play_after,
            handle_suffix=f"{handle}{handle_suffix}",
        )
        handle = measure_acquire_section.children[1].handle

        match_section = Match(
            uid=f"feedback_{qubit.uid}{handle_suffix}",
            handle=handle,
            play_after=measure_acquire_section,
        )
        if "g" in states_to_reset:
            case = Case(state=0)
            case.play(
                signal=self.signal_name("drive", qubit),
                pulse=qt_ops.quantum_gate(qubit, "X180_ge"),
                amplitude=0,
            )
            match_section.add(case)
        if "e" in states_to_reset:
            case = Case(state=1)
            case.play(
                signal=self.signal_name("drive", qubit),
                pulse=qt_ops.quantum_gate(qubit, "X180_ge"),
            )
            match_section.add(case)
        if "f" in states_to_reset:
            if len(qubit.get_integration_kernels()) < 2:
                raise NotImplementedError(
                    f"Currently active reset of levels higher than 'e' requires "
                    f"multi-state discrimination and at least 2 optimised "
                    f"integration kernels, but {qubit.uid} has fewer than 2 "
                    f"kernels."
                )
            case = Case(state=2)
            case.play(
                signal=self.signal_name("drive_ef", qubit),
                pulse=qt_ops.quantum_gate(qubit, "X180_ef"),
            )
            case.play(
                signal=self.signal_name("drive", qubit),
                pulse=qt_ops.quantum_gate(qubit, "X180_ge"),
            )
            match_section.add(case)

        return [measure_acquire_section, match_section]

    def create_transition_preparation_sections(self, qubit, **kwargs):
        if hasattr(self, "transition_to_calibrate"):
            # All children of SingleQubitGateTuneup
            return self.create_state_preparation_sections(
                qubit, state_to_prepare=self.transition_to_calibrate[0], **kwargs
            )

    def create_qubit_preparation_sections(self, qubit, **kwargs):
        preparation_sections = []
        if self.preparation_type == "active_reset":
            states_to_reset = self.experiment_metainfo.get(
                "states_to_actively_reset", ("g", "e")
            )
            preparation_sections += self.create_active_reset_sections(
                qubit, states_to_reset, **kwargs
            )
        # return None if no preparation sections were created because the output
        # of this method is intended to be passed to the play_after of another
        # section.
        return preparation_sections


class ConfigurableExperiment(StatePreparationMixin):
    fallback_experiment_name = "ConfigurableExperiment"
    valid_user_parameters = dict(
        experiment_metainfo=["preparation_type", "states_to_actively_reset"],
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
        apply_exit_condition=False,
        check_valid_user_parameters=True,
        run=False,
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
        self.preparation_type = self.experiment_metainfo.get("preparation_type", "wait")
        self.cal_states = self.experiment_metainfo.get("cal_states", None)
        if acquisition_metainfo is None:
            acquisition_metainfo = {}
        self.acquisition_metainfo = dict(count=2**12)
        # overwrite default with user-provided options
        self.acquisition_metainfo.update(acquisition_metainfo)
        if qubit_temporary_values is None:
            qubit_temporary_values = {}
        self.qubit_temporary_values = qubit_temporary_values

        self.experiment_name = experiment_name
        if self.experiment_name is None:
            self.experiment_name = self.fallback_experiment_name
        if self.preparation_type != "wait":
            self.experiment_name += f"_{self.preparation_type}"
        self.create_experiment_label()

        self.signals = signals
        if self.signals is None:
            self.signals = ["drive", "measure", "acquire"]
        (
            self.experiment_signals,
            self.experiment_signal_uids_qubit_map,
        ) = self.create_experiment_signals(self.qubits, self.signals)

        self.apply_exit_condition = apply_exit_condition
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
        Sets the measurement_setup calibration from the qubits calibrations.
        Sets the experiment calibration if preparation_type == "active_reset".

        To be overridden by children for further settings of the experiment
        calibration.

        """
        self.update_measurement_setup()
        cal = Calibration()
        if self.preparation_type == "active_reset":
            for qubit in self.qubits:
                cal[self.signal_name("acquire", qubit)] = SignalCalibration(
                    oscillator=Oscillator(
                        frequency=0, modulation_type=ModulationType.SOFTWARE
                    ),
                )
            self.experiment.set_calibration(cal)

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
                if not isinstance(sec, Case):
                    # making unique uids for the pulses in Cases results in an
                    # error
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

    def execute_exit_condition(self):
        # to be overridden by children
        pass

    def autorun(self):
        try:
            with calib_hlp.QubitTemporaryValuesContext(*self.qubit_temporary_values):
                self.define_experiment()
                self.configure_experiment()
                self.compile_experiment()
                self.run_experiment()
                if self.apply_exit_condition:
                    self.execute_exit_condition()
        except Exception:
            log.error("Unhandled error during ConfigurableExperiment!")
            log.error(traceback.format_exc())

    def create_acquire_rt_loop(self):
        self.acquire_loop = AcquireLoopRt(**self.acquisition_metainfo)

    def create_measure_acquire_sections(
        self,
        qubit,
        uid=None,
        play_after=None,
        handle_suffix=None,
        integration_kernel="default",
    ):
        handle = f"{self.experiment_name}_{qubit.uid}"
        if handle_suffix is not None:
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


class ExperimentTemplate(ConfigurableExperiment):
    fallback_experiment_name = "Experiment"
    save_directory = None
    timestamp = None
    results = None
    analysis_results = None
    valid_user_parameters = merge_valid_user_parameters(
        [
            dict(
                analysis_metainfo=[
                    "figure_name",
                    "overwrite_figures",
                    "figure_formats",
                ],
            ),
            ConfigurableExperiment.valid_user_parameters,
        ]
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
        apply_exit_condition=False,
        do_analysis=True,
        analysis_metainfo=None,
        save=True,
        data_directory=None,
        update=False,
        run=False,
        check_valid_user_parameters=True,
        **kwargs,
    ):
        self.data_directory = data_directory
        self.do_analysis = do_analysis
        self.analysis_metainfo = analysis_metainfo
        if self.analysis_metainfo is None:
            self.analysis_metainfo = {}
        self.update = update
        if not self.do_analysis:
            self.update = False
        self.save = save
        if self.save and self.data_directory is None:
            raise ValueError(
                "save==True, but no data_directory was specified. "
                "Please provide data_directory or set save=False."
            )

        super().__init__(
            qubits,
            session,
            measurement_setup,
            experiment_name,
            signals,
            sweep_parameters_dict,
            experiment_metainfo,
            acquisition_metainfo,
            qubit_temporary_values,
            apply_exit_condition,
            check_valid_user_parameters,
            run=False,
            **kwargs,
        )

        # generate_timestamp_save_directory requires the experiment label
        # created in the super call
        self.generate_timestamp_save_directory()
        self.run = run
        if self.run:
            self.autorun()

    def analyse_experiment(self):
        # to be overridden by children
        self.analysis_results = {
            qubit.uid: dict(
                new_parameter_values=dict(),
                old_parameter_values=dict(),
                fit_results=None,
            )
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

    def reset_setup(self):
        old_qb_params_exist = any(
            [
                len(self.analysis_results[qubit.uid]["old_parameter_values"]) > 0
                for qubit in self.qubits
            ]
        )
        if old_qb_params_exist:
            for qubit in self.qubits:
                old_qb_pars = self.analysis_results[qubit.uid]["old_parameter_values"]
                for qb_par, par_value in old_qb_pars.items():
                    if getattr(qubit.parameters, qb_par, None) is not None:
                        setattr(qubit.parameters, qb_par, par_value)
                        if qb_par == "dc_voltage_parking":
                            nt_cb_func = self.experiment_metainfo.get(
                                "neartime_callback_function", None
                            )
                            if nt_cb_func is None:
                                raise ValueError(
                                    "Please provide the neartime callback function "
                                    "for setting the dc voltage in "
                                    "experiment_metainfo['neartime_callback_function']."
                                )
                            log.info(
                                f"Updating DC voltage source slot "
                                f"{qubit.parameters.dc_slot} ({qubit.uid}) to the "
                                f"new value of {par_value:.4f}."
                            )
                            nt_cb_func(self.session, par_value, qubit)
                    elif qb_par.startswith("ge"):
                        par_name = qb_par.split("_")[-1]
                        qubit.parameters.drive_parameters_ge[par_name] = par_value
                    elif qb_par.startswith("ef"):
                        par_name = qb_par.split("_")[-1]
                        qubit.parameters.drive_parameters_ef[par_name] = par_value
                    else:
                        log.warning(
                            f"Parameter {qb_par} was not found for "
                            f"{qubit.uid}. This parameter was not reset."
                        )
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
        def get_filepath(filename):
            filename_full = f"{self.timestamp}_{filename}"
            return os.path.abspath(os.path.join(self.save_directory, filename_full))

        # Save as json using the LabOneQ serialiser
        filename = "measurement_setup_before_experiment.json"
        if filename not in os.listdir(self.save_directory):
            # only save the setup if the file does not already exist
            self.measurement_setup.save(get_filepath(filename))

        # Save as pickle for fallback
        filename = "measurement_setup_before_experiment.p"
        if filename not in os.listdir(self.save_directory):
            with open(get_filepath(filename), "wb") as f:
                pickle.dump(self.measurement_setup, f)

    def save_experiment_metainfo(self):
        # Save the meta-information
        exp_metainfo = {
            "experiment_name": self.experiment_name,
            "experiment_label": self.experiment_label,
            "sweep_parameters_dict": self.sweep_parameters_dict,
            "analysis_metainfo": self.analysis_metainfo,
            "timestamp": self.timestamp,
            "save_directory": self.save_directory,
        }
        exp_metainfo.update(self.experiment_metainfo)
        metainfo_file = os.path.abspath(
            os.path.join(self.save_directory, f"{self.timestamp}_meta_information.p")
        )
        with open(metainfo_file, "wb") as f:
            pickle.dump(exp_metainfo, f)

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

            # Save only the acquired_results as pickle: fallback in case
            # something goes wrong with the deserialisation of Results.
            # AnalysisResults is less likely to change between sprints
            filename = os.path.abspath(
                os.path.join(
                    self.save_directory,
                    f"{self.timestamp}_acquired_results{filename_suffix}.json",
                )
            )
            from laboneq.dsl.serialization import Serializer
            Serializer.to_json_file(self.results.acquired_results, filename)

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
        old_qb_params_exist = any(
            [
                len(self.analysis_results[qubit.uid]["old_parameter_values"]) > 0
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

        if (
            new_qb_params_exist
            or old_qb_params_exist
            or fit_results_exist
            or other_ana_res_exist
        ):
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
                super().autorun()
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
            log.error("Unhandled error during ExperimentTemplate!")
            log.error(traceback.format_exc())
