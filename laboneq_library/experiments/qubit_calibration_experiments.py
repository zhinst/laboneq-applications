import numpy as np
from copy import deepcopy
import uncertainties as unc
import matplotlib.pyplot as plt
from ruamel.yaml import YAML

ryaml = YAML()

import logging

log = logging.getLogger(__name__)

from . import quantum_operations as qt_ops
from laboneq.analysis import fitting as fit_mods
from laboneq.simple import *  # noqa: F403
from laboneq_library.analysis import analysis_helpers as ana_hlp
from laboneq_library.experiments.experiment_library import (
    ExperimentTemplate,
    merge_valid_user_parameters,
)


class QubitSpectroscopy(ExperimentTemplate):
    fallback_experiment_name = "QubitSpectroscopy"
    valid_user_parameters = merge_valid_user_parameters(
        [
            dict(
                experiment_metainfo=[
                    "neartime_sweep_parameter",
                    "neartime_callback_function",
                    "pulsed",
                ],
                analysis_metainfo=[
                    "frequency_filter_for_fit",
                    "find_peaks",
                    "do_fitting",
                    "param_hints",
                    "show_figures",
                ],
            ),
            ExperimentTemplate.valid_user_parameters,
        ]
    )

    def __init__(self, *args, **kwargs):
        experiment_metainfo = kwargs.get("experiment_metainfo", dict())
        self.nt_swp_par = experiment_metainfo.get("neartime_sweep_parameter", None)
        self.pulsed = experiment_metainfo.get("pulsed", True)
        if not self.pulsed:
            raise NotImplementedError(
                "Continuous-wave qubit spectroscopy is currently not implemented."
            )
        # Add suffix to experiment name
        experiment_name = kwargs.get("experiment_name", self.fallback_experiment_name)
        if self.nt_swp_par is not None:
            experiment_name += f"{self.nt_swp_par[0].upper()}{self.nt_swp_par[1:]}Sweep"
        experiment_name += "_Pulsed" if self.pulsed else "_CW"
        kwargs["experiment_name"] = experiment_name
        run = kwargs.pop("run", False)  # instantiate base without running exp
        kwargs["run"] = False
        super().__init__(*args, **kwargs)

        for qubit in self.qubits:
            freq_swp = self.sweep_parameters_dict[qubit.uid][0]
            if all(freq_swp.values > 1e9):
                # sweep values are passed as qubit resonance frequencies:
                # subtract lo freq to sweep if freq
                if_freq_swp = SweepParameter(
                    f"if_freq_{qubit.uid}",
                    values=freq_swp.values - qubit.parameters.drive_lo_frequency,
                    axis_name=freq_swp.axis_name,
                    driven_by=[freq_swp],
                )
                self.sweep_parameters_dict[qubit.uid][0] = if_freq_swp

        self.run = run
        if self.run:
            self.autorun()

    def define_experiment(self):
        self.experiment.sections = []
        self.create_acquire_rt_loop()
        if self.nt_swp_par == "frequency":
            nt_freq_sweep = Sweep(
                uid="neartime_frequency_sweep",
                parameters=[
                    self.sweep_parameters_dict[qubit.uid][1] for qubit in self.qubits
                ],
            )
            # add near-time sweep to experiment
            self.experiment.add(nt_freq_sweep)
            # add real-time loop to near-time sweep
            nt_freq_sweep.add(self.acquire_loop)
        else:
            self.experiment.add(self.acquire_loop)

        for i, qubit in enumerate(self.qubits):
            # create a freq_sweep for each qubit
            freq_sweep = Sweep(
                uid=f"frequency_sweep_{qubit.uid}",
                parameters=[self.sweep_parameters_dict[qubit.uid][0]],
            )
            spec_pulse_amp = None
            if self.nt_swp_par == "amplitude":
                spec_pulse_amp = self.sweep_parameters_dict[qubit.uid][1]
                amp_sweep = Sweep(
                    uid=f"amplitude_sweep_{qubit.uid}", parameters=[spec_pulse_amp]
                )
                amp_sweep.add(freq_sweep)
                self.acquire_loop.add(amp_sweep)
            elif self.nt_swp_par == "voltage":
                voltage_sweep_par = self.sweep_parameters_dict[qubit.uid][1]
                voltage_sweep = Sweep(
                    uid=f"neartime_{self.nt_swp_par}_sweep_{qubit.uid}",
                    parameters=[voltage_sweep_par],
                )
                ntsf = self.experiment_metainfo.get("neartime_callback_function", None)
                if ntsf is None:
                    raise ValueError(
                        "Please provide the neartime callback function for "
                        "the voltage sweep in "
                        "experiment_metainfo['neartime_callback_function']."
                    )
                # all near-time callback functions have the format
                # func(session, sweep_param_value, qubit)
                voltage_sweep.call(ntsf, voltage=voltage_sweep_par, qubit=qubit)
                if i == 0:
                    # voltage sweep cannot be parallelised at the moment
                    # remove the acquire_loop from the experiment
                    self.experiment.sections = []
                self.experiment.add(voltage_sweep)
                voltage_sweep.add(self.acquire_loop)
                self.acquire_loop.add(freq_sweep)
            else:
                self.acquire_loop.add(freq_sweep)

            integration_kernel = None
            if self.pulsed:
                excitation_section = Section(uid=f"{qubit.uid}_excitation")
                spec_pulse = pulse_library.const(
                    uid=f"spectroscopy_pulse_{qubit.uid}",
                    length=qubit.parameters.spectroscopy_pulse_length,
                    amplitude=qubit.parameters.spectroscopy_amplitude,
                    can_compress=True,  # fails without this!
                )
                integration_kernel = pulse_library.const(
                    length=qubit.parameters.readout_integration_length,
                    amplitude=1,
                )
                excitation_section.play(
                    signal=self.signal_name("drive", qubit),
                    pulse=spec_pulse,
                    amplitude=spec_pulse_amp,
                )
                freq_sweep.add(excitation_section)

            measure_section = self.create_measure_acquire_sections(
                qubit=qubit,
                integration_kernel=integration_kernel,
                play_after=f"{qubit.uid}_excitation" if self.pulsed else None,
            )

            freq_sweep.add(measure_section)

    def configure_experiment(self):
        super().configure_experiment()

        cal = self.experiment.get_calibration()
        for qubit in self.qubits:
            qb_sweep = self.sweep_parameters_dict[qubit.uid]
            local_oscillator = None
            drive_amplitude = None
            if self.nt_swp_par == "amplitude" and not self.pulsed:
                drive_amplitude = qb_sweep[1]
            elif self.nt_swp_par == "frequency":
                local_oscillator = Oscillator(frequency=qb_sweep[1])

            freq_swp = self.sweep_parameters_dict[qubit.uid][0]
            sig_calib_kwargs = {}
            if local_oscillator is not None:
                sig_calib_kwargs["local_oscillator"] = local_oscillator
            if drive_amplitude is not None:
                sig_calib_kwargs["amplitude"] = drive_amplitude
            cal[self.signal_name("drive", qubit)] = SignalCalibration(
                oscillator=Oscillator(
                    frequency=freq_swp, modulation_type=ModulationType.HARDWARE
                ),
                **sig_calib_kwargs,
            )
        self.experiment.set_calibration(cal)

    def analyse_experiment(self):
        super().analyse_experiment()
        freq_filter = self.analysis_metainfo.get("frequency_filter_for_fit", {})
        if not hasattr(freq_filter, "__iter__"):
            freq_filter = {qubit.uid: freq_filter for qubit in self.qubits}
        find_peaks = self.analysis_metainfo.get("find_peaks", {})
        if not hasattr(find_peaks, "__iter__"):
            find_peaks = {qubit.uid: find_peaks for qubit in self.qubits}
        for qubit in self.qubits:
            new_parameter_values = self.analysis_results[qubit.uid][
                "new_parameter_values"
            ]
            old_parameter_values = self.analysis_results[qubit.uid][
                "old_parameter_values"
            ]
            # get frequency filter of qubit
            ff_qb = freq_filter.get(qubit.uid, None)
            # extract data
            handle = f"{self.experiment_name}_{qubit.uid}"
            data_mag = abs(self.results.get_data(handle))
            res_axis = self.results.get_axis(handle)
            if self.nt_swp_par is None or self.nt_swp_par == "frequency":
                data_mag = np.array([data for data in data_mag]).flatten()
                if len(res_axis) > 1:
                    outer = self.results.get_axis(handle)[0]
                    if isinstance(outer, list):
                        # happens when nt_swp_par = 'frequency'
                        outer = outer[0]
                    inner = self.results.get_axis(handle)[1]
                    freqs = np.array([out + inner for out in outer]).flatten()
                else:
                    freqs = (
                        self.results.get_axis(handle)[0]
                        + qubit.parameters.drive_lo_frequency
                    )

                # plot data
                fig, ax = plt.subplots()
                ax.plot(freqs / 1e9, data_mag, "o")
                ax.set_xlabel("Qubit Frequency, $f_{\\mathrm{QB}}$ (GHz)")
                ax.set_ylabel("Signal Magnitude, $|S_{21}|$ (a.u.)")

                if self.analysis_metainfo.get("do_fitting", True):
                    data_to_fit = data_mag if ff_qb is None else data_mag[ff_qb(freqs)]
                    freqs_to_fit = freqs if ff_qb is None else freqs[ff_qb(freqs)]
                    # fit data
                    param_hints = self.analysis_metainfo.get("param_hints")
                    if param_hints is None:
                        width_guess = 50e3
                        # fit with guess values for a peak
                        param_hints = {
                            "amplitude": {"value": np.max(data_to_fit) * width_guess},
                            "position": {"value": freqs_to_fit[np.argmax(data_to_fit)]},
                            "width": {"value": width_guess},
                            "offset": {"value": 0},
                        }
                        fit_res_peak = ana_hlp.fit_data_lmfit(
                            fit_mods.lorentzian,
                            freqs_to_fit,
                            data_to_fit,
                            param_hints=param_hints,
                        )
                        # fit with guess values for a dip
                        param_hints["amplitude"]["value"] *= -1
                        param_hints["position"]["value"] = freqs_to_fit[
                            np.argmin(data_to_fit)
                        ]
                        fit_res_dip = ana_hlp.fit_data_lmfit(
                            fit_mods.lorentzian,
                            freqs_to_fit,
                            data_to_fit,
                            param_hints=param_hints,
                        )
                        # determine whether there is a peak or a dip: compare
                        # the distance between the value at the fitted peak/dip
                        # to the mean of the data_mag array: the larger distance
                        # is the true spectroscopy signal
                        dpeak = abs(
                            fit_res_peak.model.func(
                                fit_res_peak.best_values["position"],
                                **fit_res_peak.best_values,
                            )
                            - np.mean(data_to_fit)
                        )
                        ddip = abs(
                            fit_res_dip.model.func(
                                fit_res_dip.best_values["position"],
                                **fit_res_dip.best_values,
                            )
                            - np.mean(data_to_fit)
                        )
                        fit_res = fit_res_peak if dpeak > ddip else fit_res_dip
                    else:
                        # do what the user asked
                        fit_res = ana_hlp.fit_data_lmfit(
                            fit_mods.lorentzian,
                            freqs_to_fit,
                            data_to_fit,
                            param_hints=param_hints,
                        )
                    self.analysis_results[qubit.uid]["fit_results"] = fit_res
                    fqb = fit_res.params["position"].value
                    fqb_err = fit_res.params["position"].stderr
                    new_parameter_values.update({"resonance_frequency_ge": fqb})
                    old_parameter_values.update(
                        {
                            "resonance_frequency_ge": qubit.parameters.resonance_frequency_ge
                        }
                    )

                    # plot fit
                    freqs_fine = np.linspace(freqs_to_fit[0], freqs_to_fit[-1], 501)
                    ax.plot(
                        freqs_fine / 1e9,
                        fit_res.model.func(freqs_fine, **fit_res.best_values),
                        "r-",
                    )
                    textstr = (
                        f"Extracted qubit frequency: {fqb / 1e9:.4f} GHz "
                        f"$\\pm$ {fqb_err / 1e9:.4f} GHz"
                    )
                    textstr += (
                        f"\nOld qubit frequency: "
                        f"{qubit.parameters.resonance_frequency_ge / 1e9:.4f} GHz"
                    )
                    ax.text(
                        0, -0.15, textstr, ha="left", va="top", transform=ax.transAxes
                    )
            else:
                # 2D plot of results
                nt_sweep_par_vals = self.results.get_axis(handle)[0]
                nt_sweep_par_name = self.results.get_axis_name(handle)[0]
                freqs = (
                    self.results.get_axis(handle)[1]
                    + qubit.parameters.drive_lo_frequency
                )
                data_mag = abs(self.results.get_data(handle))

                X, Y = np.meshgrid(freqs / 1e9, nt_sweep_par_vals)
                fig, ax = plt.subplots(constrained_layout=True)

                CS = ax.contourf(X, Y, data_mag, levels=100, cmap="magma")
                ax.set_title(f"{handle}")
                ax.set_xlabel("Qubit Frequency, $f_{\\mathrm{QB}}$ (GHz)")
                ax.set_ylabel(nt_sweep_par_name)
                cbar = fig.colorbar(CS)
                cbar.set_label("Signal Magnitude, $|S_{21}|$ (a.u.)")

                if self.nt_swp_par == "voltage":
                    # 1D plot of the qubit frequency vs voltage
                    # decide whether to extract peaks or dips for qubit
                    fp_qb = find_peaks.get(qubit.uid, True)
                    take_extremum = np.argmax if fp_qb else np.argmin
                    if ff_qb is None:
                        freqs_peaks = freqs[take_extremum(data_mag, axis=1)]
                    else:
                        mask = ff_qb(freqs)
                        print(len(freqs), len(freqs[mask]))
                        freqs_peaks = freqs[mask][
                            take_extremum(data_mag[:, mask], axis=1)
                        ]
                    # plot
                    ax.plot(freqs_peaks / 1e9, nt_sweep_par_vals, "ow")
                    # figure out whether voltages vs freqs is convex or concave
                    take_extremum_fit, scf = (
                        (np.argmax, 1)
                        if (ana_hlp.is_data_convex(nt_sweep_par_vals, freqs_peaks))
                        else (np.argmin, -1)
                    )
                    # optimal parking parameters at the extremum of
                    # voltages vs frequencies
                    f0 = freqs_peaks[take_extremum_fit(freqs_peaks)]
                    V0 = nt_sweep_par_vals[take_extremum_fit(freqs_peaks)]
                    new_parameter_values.update(
                        {"resonance_frequency_ge": f0, "dc_voltage_parking": V0}
                    )
                    old_parameter_values.update(
                        {
                            "resonance_frequency_ge": qubit.parameters.resonance_frequency_ge,
                            "dc_voltage_parking": qubit.parameters.dc_voltage_parking,
                        }
                    )

                    if self.analysis_metainfo.get("do_fitting", True):
                        # fit frequency vs voltage and take the optimal parking
                        # parameters from fit
                        param_hints = self.analysis_metainfo.get(
                            "param_hints",
                            {
                                "voltage_sweet_spot": {"value": V0},
                                "frequency_sweet_spot": {"value": f0},
                                "frequency_voltage_scaling": {
                                    "value": scf * (max(freqs_peaks) - min(freqs_peaks))
                                },
                            },
                        )
                        fit_res = ana_hlp.fit_data_lmfit(
                            fit_mods.transmon_voltage_dependence_quadratic,
                            nt_sweep_par_vals,
                            freqs_peaks,
                            param_hints=param_hints,
                        )
                        self.analysis_results[qubit.uid]["fit_results"] = fit_res
                        new_parameter_values.update(
                            {
                                "resonance_frequency_ge": fit_res.best_values[
                                    "frequency_sweet_spot"
                                ],
                                "dc_voltage_parking": fit_res.best_values[
                                    "voltage_sweet_spot"
                                ],
                            }
                        )
                        # plot fit
                        ntpval_fine = np.linspace(
                            nt_sweep_par_vals[0], nt_sweep_par_vals[-1], 501
                        )
                        ax.plot(
                            fit_res.model.func(ntpval_fine, **fit_res.best_values)
                            / 1e9,
                            ntpval_fine,
                            "w-",
                        )
                        f0 = fit_res.best_values["frequency_sweet_spot"]
                        f0err = fit_res.params["frequency_sweet_spot"].stderr
                        V0 = fit_res.best_values["voltage_sweet_spot"]
                        V0err = fit_res.params["voltage_sweet_spot"].stderr
                        ax.plot(
                            f0 / 1e9,
                            V0,
                            "sC2",
                            markersize=plt.rcParams["lines.markersize"] + 1,
                        )
                        textstr = f"Parking voltage: {V0:.4f} $\\pm$ {V0err:.4f} V"
                        textstr += f"\nParking frequency: {f0 / 1e9:.4f} $\\pm$ {f0err / 1e9:.4f} GHz"
                        ax.text(
                            0,
                            -0.15,
                            textstr,
                            ha="left",
                            va="top",
                            transform=ax.transAxes,
                        )

            ax.set_title(f"{self.timestamp}_{handle}")
            # save figures and results
            if self.save:
                # Save the figure
                self.save_figure(fig, qubit)
            if self.analysis_metainfo.get("show_figures", False):
                plt.show()
            plt.close(fig)

    def execute_exit_condition(self):
        if self.nt_swp_par == "voltage":
            # set the dc voltage source to the original voltage value stored in the
            # qubit parameters
            ntsf = self.experiment_metainfo.get("neartime_callback_function", None)
            if ntsf is None:
                raise ValueError(
                    "Please provide the neartime callback function for the voltage"
                    "sweep in experiment_metainfo['neartime_callback_function']."
                )
            for qubit in self.qubits:
                log.info(
                    f"Setting DC voltage source slot "
                    f"{qubit.parameters.dc_slot} ({qubit.uid}) back to the "
                    f"original value of {qubit.parameters.dc_voltage_parking:.4f}."
                )
                ntsf(self.session, qubit.parameters.dc_voltage_parking, qubit)

    def update_qubit_parameters(self):
        for qubit in self.qubits:
            new_qb_pars = self.analysis_results[qubit.uid]["new_parameter_values"]
            if len(new_qb_pars) == 0:
                return

            qubit.parameters.resonance_frequency_ge = new_qb_pars[
                "resonance_frequency_ge"
            ]
            if "dc_voltage_parking" in new_qb_pars:
                qubit.parameters.dc_voltage_parking = new_qb_pars["dc_voltage_parking"]

    def update_entire_setup(self):
        super().update_entire_setup()
        if self.nt_swp_par == "voltage":
            # set the dc voltage source to the new voltage value
            nt_cb_func = self.experiment_metainfo.get(
                "neartime_callback_function", None
            )
            if nt_cb_func is None:
                raise ValueError(
                    "Please provide the neartime callback function for setting the "
                    "dc voltage in experiment_metainfo['neartime_callback_function']."
                )
            for qubit in self.qubits:
                nt_cb_func(self.session, qubit.parameters.dc_voltage_parking, qubit)


######################################################
#### Single-Qubit Gate Tune-up Experiment classes ####
######################################################


class SingleQubitGateTuneup(ExperimentTemplate):
    valid_user_parameters = merge_valid_user_parameters(
        [
            dict(
                experiment_metainfo=["cal_states", "transition_to_calib"],
                analysis_metainfo=[
                    "do_pca",
                    "show_figures",
                ],
            ),
            ExperimentTemplate.valid_user_parameters,
        ]
    )

    def __init__(self, *args, signals=None, **kwargs):
        exp_metainfo = kwargs.get("experiment_metainfo", {})
        self.transition_to_calib = exp_metainfo.get("transition_to_calib", "ge")

        # Add "f" to cal states if transition_to_calib contains f
        cal_states = exp_metainfo.get("cal_states", None)
        if cal_states is None:
            cal_states = "gef" if "f" in self.transition_to_calib else "ge"
        exp_metainfo["cal_states"] = cal_states
        kwargs["experiment_metainfo"] = exp_metainfo

        # Add drive_ef to signals if not there
        if signals is None:
            signals = ["drive", "measure", "acquire"]
        if "f" in self.transition_to_calib and "drive_ef" not in signals:
            signals += ["drive_ef"]

        # Suffix of the drive signal
        self.drive_signal_suffix = "_ef" if self.transition_to_calib == "ef" else ""

        # Add suffix to experiment name
        experiment_name = kwargs.get("experiment_name", self.fallback_experiment_name)
        experiment_name += f"_{self.transition_to_calib}"
        kwargs["experiment_name"] = experiment_name

        super().__init__(*args, signals=signals, **kwargs)

    def create_qubit_preparation_sections(self, qubit, play_after=None, **kwargs):
        if play_after is None:
            play_after = []
        elif not hasattr(play_after, "__iter__"):
            play_after = [play_after]

        qb_prep_sections = super().create_qubit_preparation_sections(
            qubit, play_after=play_after, **kwargs
        )
        transition_prep_sections = self.create_transition_preparation_sections(
            qubit, play_after=play_after + qb_prep_sections
        )
        return qb_prep_sections + transition_prep_sections

    def analyse_experiment(self):
        super().analyse_experiment()
        for qubit in self.qubits:
            # extract data
            handle = f"{self.experiment_name}_{qubit.uid}"
            do_pca = self.analysis_metainfo.get("do_pca", False)
            data_dict = ana_hlp.extract_and_rotate_data_1d(
                self.results, handle, cal_states=self.cal_states, do_pca=do_pca
            )
            num_cal_traces = data_dict["num_cal_traces"]
            self.analysis_results[qubit.uid]["rotated_data"] = data_dict

            # configure plot: data is plotted in analyse_experiment_qubit
            fig, ax = plt.subplots()
            ax.set_xlabel(self.results.get_axis_name(handle)[0])
            ax.set_ylabel(
                "Principal Component (a.u)"
                if (num_cal_traces == 0 or do_pca)
                else f"$|{self.cal_states[-1]}\\rangle$-State Population"
            )
            ax.set_title(f"{self.timestamp}_{handle}")
            # run the analysis from the children
            self.analyse_experiment_qubit(qubit, data_dict, fig, ax)
            if self.save:
                # Save the figure
                self.save_figure(fig, qubit)
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
    valid_user_parameters = merge_valid_user_parameters(
        [
            dict(
                analysis_metainfo=[
                    "do_fitting",
                    "param_hints",
                ],
            ),
            SingleQubitGateTuneup.valid_user_parameters,
        ]
    )

    def define_experiment(self):
        self.experiment.sections = []
        # define Rabi experiment pulse sequence
        # outer loop - real-time, cyclic averaging
        self.create_acquire_rt_loop()
        self.experiment.add(self.acquire_loop)
        for i, qubit in enumerate(self.qubits):
            # create sweep
            sweep = Sweep(
                uid=f"{qubit.uid}_{self.experiment_name}_sweep",
                parameters=[self.sweep_parameters_dict[qubit.uid][0]],
            )

            # create preparation pulses sections
            prep_sections = self.create_qubit_preparation_sections(qubit)

            # create pulses section
            excitation_section = Section(
                uid=f"{qubit.uid}_excitation",
                alignment=SectionAlignment.LEFT,
                on_system_grid=True,
                play_after=prep_sections,
            )
            # pulse to calibrate
            drive_pulse = qt_ops.quantum_gate(qubit, f"X180_{self.transition_to_calib}")
            # amplitude is scaled w.r.t this value
            drive_pulse.amplitude = 1
            excitation_section.play(
                signal=self.signal_name(f"drive{self.drive_signal_suffix}", qubit),
                pulse=drive_pulse,
                amplitude=self.sweep_parameters_dict[qubit.uid][0],
            )

            # create readout + acquire section
            measure_section = self.create_measure_acquire_sections(
                qubit=qubit,
                play_after=f"{qubit.uid}_excitation",
            )

            # add sweep and sections to acquire loop rt
            self.acquire_loop.add(sweep)
            for prep_sec in prep_sections:
                sweep.add(prep_sec)
            sweep.add(excitation_section)
            sweep.add(measure_section)
            self.add_cal_states_sections(qubit, add_to=self.acquire_loop)

    def analyse_experiment_qubit(self, qubit, data_dict, figure, ax):
        # plot data
        ax.plot(
            data_dict["sweep_points_w_cal_traces"],
            data_dict["data_rotated_w_cal_traces"],
            "o",
            zorder=2,
        )
        if self.analysis_metainfo.get("do_fitting", True):
            swpts_to_fit = data_dict["sweep_points"]
            data_to_fit = data_dict["data_rotated"]
            # fit data
            freqs_guess, phase_guess = ana_hlp.find_oscillation_frequency_and_phase(
                data_to_fit, swpts_to_fit
            )
            param_hints = self.analysis_metainfo.get(
                "param_hints",
                {
                    "frequency": {"value": 2 * np.pi * freqs_guess, "min": 0},
                    "phase": {"value": phase_guess},
                    "amplitude": {
                        "value": abs(max(data_to_fit) - min(data_to_fit)) / 2,
                        "min": 0,
                    },
                    "offset": {"value": np.mean(data_to_fit)},
                },
            )
            fit_res = ana_hlp.fit_data_lmfit(
                fit_mods.oscillatory, swpts_to_fit, data_to_fit, param_hints=param_hints
            )
            self.analysis_results[qubit.uid]["fit_results"] = fit_res

            freq_fit = unc.ufloat(
                fit_res.params["frequency"].value, fit_res.params["frequency"].stderr
            )
            phase_fit = unc.ufloat(
                fit_res.params["phase"].value, fit_res.params["phase"].stderr
            )
            (
                pi_amps_top,
                pi_amps_bottom,
                pi2_amps_rise,
                pi2_amps_fall,
            ) = ana_hlp.get_pi_pi2_xvalues_on_cos(swpts_to_fit, freq_fit, phase_fit)
            # if pca is done, it can happen that the pi-pulse amplitude
            # is in pi_amps_bottom and the pi/2-pulse amplitude in pi2_amps_fall
            pi_amps = np.sort(np.concatenate([pi_amps_top, pi_amps_bottom]))
            pi2_amps = np.sort(np.concatenate([pi2_amps_rise, pi2_amps_fall]))
            pi2_amp = pi2_amps[0]
            pi_amp = pi_amps[pi_amps > pi2_amp][0]
            self.analysis_results[qubit.uid]["new_parameter_values"].update(
                {
                    f"{self.transition_to_calib}_amplitude_pi": pi_amp.nominal_value,
                    f"{self.transition_to_calib}_amplitude_pi2": pi2_amp.nominal_value,
                    f"{self.transition_to_calib}_pi_amps": [
                        pia.nominal_value for pia in pi_amps
                    ],
                    f"{self.transition_to_calib}_pi2_amps": [
                        pi2a.nominal_value for pi2a in pi_amps
                    ],
                }
            )

            # plot fit
            swpts_fine = np.linspace(swpts_to_fit[0], swpts_to_fit[-1], 501)
            ax.plot(
                swpts_fine,
                fit_res.model.func(swpts_fine, **fit_res.best_values),
                "r-",
                zorder=1,
            )
            plt.plot(
                pi_amp.nominal_value,
                fit_res.model.func(pi_amp.nominal_value, **fit_res.best_values),
                "sk",
                zorder=3,
                markersize=plt.rcParams["lines.markersize"] + 1,
            )
            plt.plot(
                pi2_amp.nominal_value,
                fit_res.model.func(pi2_amp.nominal_value, **fit_res.best_values),
                "sk",
                zorder=3,
                markersize=plt.rcParams["lines.markersize"] + 1,
            )
            # textbox
            old_pi_amp = (
                qubit.parameters.drive_parameters_ef["amplitude_pi"]
                if "f" in self.transition_to_calib
                else qubit.parameters.drive_parameters_ge["amplitude_pi"]
            )
            old_pi2_amp = (
                qubit.parameters.drive_parameters_ef["amplitude_pi2"]
                if "f" in self.transition_to_calib
                else qubit.parameters.drive_parameters_ge["amplitude_pi2"]
            )
            self.analysis_results[qubit.uid]["old_parameter_values"].update(
                {
                    f"{self.transition_to_calib}_amplitude_pi": old_pi_amp,
                    f"{self.transition_to_calib}_amplitude_pi2": old_pi2_amp,
                }
            )
            textstr = (
                "$A_{\\pi}$: "
                + f"{pi_amp.nominal_value:.4f} $\\pm$ {pi_amp.std_dev:.4f}"
            )
            textstr += "\nOld $A_{\\pi}$: " + f"{old_pi_amp:.4f}"
            ax.text(0, -0.15, textstr, ha="left", va="top", transform=ax.transAxes)
            textstr = (
                "$A_{\\pi/2}$: "
                + f"{pi2_amp.nominal_value:.4f} $\\pm$ {pi2_amp.std_dev:.4f}"
            )
            textstr += "\nOld $A_{\\pi/2}$: " + f"{old_pi2_amp:.4f}"
            ax.text(0.69, -0.15, textstr, ha="left", va="top", transform=ax.transAxes)

    def update_qubit_parameters(self):
        for qubit in self.qubits:
            new_qb_pars = self.analysis_results[qubit.uid]["new_parameter_values"]
            if len(new_qb_pars) == 0:
                return

            dr_pars = (
                qubit.parameters.drive_parameters_ef
                if "f" in self.transition_to_calib
                else qubit.parameters.drive_parameters_ge
            )
            dr_pars["amplitude_pi"] = new_qb_pars[
                f"{self.transition_to_calib}_amplitude_pi"
            ]
            dr_pars["amplitude_pi2"] = new_qb_pars[
                f"{self.transition_to_calib}_amplitude_pi2"
            ]


class Ramsey(SingleQubitGateTuneup):
    fallback_experiment_name = "Ramsey"
    valid_user_parameters = merge_valid_user_parameters(
        [
            dict(
                experiment_metainfo=["detuning"],
                analysis_metainfo=[
                    "do_fitting",
                    "param_hints",
                ],
            ),
            SingleQubitGateTuneup.valid_user_parameters,
        ]
    )

    def define_experiment(self):
        self.experiment.sections = []
        self.create_acquire_rt_loop()
        self.experiment.add(self.acquire_loop)
        # from the delays sweep parameters, create sweep parameters for
        # half the total delay time and for the phase of the second X90 pulse
        detuning = self.experiment_metainfo.get("detuning")
        if detuning is None:
            raise ValueError("Please provide detuning in experiment_metainfo.")
        swp_pars_phases = []
        for qubit in self.qubits:
            delays = deepcopy(self.sweep_parameters_dict[qubit.uid][0].values)
            swp_pars_phases += [
                SweepParameter(
                    uid=f"x90_phases_{qubit.uid}",
                    values=((delays - delays[0]) * detuning[qubit.uid] * 2 * np.pi)
                    % (2 * np.pi),
                )
            ]
        swp_pars_delays = [
            self.sweep_parameters_dict[qubit.uid][0] for qubit in self.qubits
        ]

        # create joint sweep for all qubits
        sweep = Sweep(
            uid=f"{self.experiment_name}_sweep",
            parameters=swp_pars_delays + swp_pars_phases,
        )
        self.acquire_loop.add(sweep)
        for i, qubit in enumerate(self.qubits):
            # create preparation pulses sections
            prep_sections = self.create_qubit_preparation_sections(qubit)

            # create ramsey-pulses section
            excitation_section = Section(
                uid=f"{qubit.uid}_excitation",
                alignment=SectionAlignment.RIGHT,
                on_system_grid=True,
                play_after=prep_sections,
            )
            # Ramsey pulses
            ramsey_drive_pulse = qt_ops.quantum_gate(
                qubit, f"X90_{self.transition_to_calib}"
            )
            excitation_section.play(
                signal=self.signal_name(f"drive{self.drive_signal_suffix}", qubit),
                pulse=ramsey_drive_pulse,
            )
            excitation_section.delay(
                signal=self.signal_name(f"drive{self.drive_signal_suffix}", qubit),
                time=swp_pars_delays[i],
            )
            excitation_section.play(
                signal=self.signal_name(f"drive{self.drive_signal_suffix}", qubit),
                pulse=ramsey_drive_pulse,
                phase=swp_pars_phases[i],
            )

            # create readout + acquire sections
            measure_section = self.create_measure_acquire_sections(
                qubit=qubit,
                play_after=f"{qubit.uid}_excitation",
            )

            # add sweep and sections to acquire loop rt
            for prep_sec in prep_sections:
                sweep.add(prep_sec)
            sweep.add(excitation_section)
            sweep.add(measure_section)
            self.add_cal_states_sections(qubit, add_to=self.acquire_loop)

    def analyse_experiment_qubit(self, qubit, data_dict, figure, ax):
        # plot data with correct scaling
        ax.plot(
            (data_dict["sweep_points_w_cal_traces"]) * 1e6,
            data_dict["data_rotated_w_cal_traces"],
            "o",
            zorder=2,
        )
        ax.set_xlabel("Pulse Separation, $\\tau$ ($\\mu$s)")
        if self.analysis_metainfo.get("do_fitting", True):
            swpts_to_fit = data_dict["sweep_points"]
            data_to_fit = data_dict["data_rotated"]
            # fit data
            freqs_guess, phase_guess = ana_hlp.find_oscillation_frequency_and_phase(
                data_to_fit, swpts_to_fit
            )
            param_hints = self.analysis_metainfo.get(
                "param_hints",
                {
                    "frequency": {"value": freqs_guess},
                    "phase": {"value": phase_guess},
                    "decay_time": {"value": 2 / 3 * max(swpts_to_fit), "min": 0},
                    "amplitude": {"value": 0.5, "vary": False},
                    "oscillation_offset": {"value": 0, "vary": "f" in self.cal_states},
                    "exponential_offset": {"value": np.mean(data_to_fit)},
                    "decay_exponent": {"value": 1, "vary": False},
                },
            )
            fit_res = ana_hlp.fit_data_lmfit(
                fit_mods.oscillatory_decay_flexible,
                swpts_to_fit,
                data_to_fit,
                param_hints=param_hints,
            )
            self.analysis_results[qubit.uid]["fit_results"] = fit_res

            t2_star = fit_res.best_values["decay_time"]
            t2_star_err = fit_res.params["decay_time"].stderr
            freq_fit = fit_res.best_values["frequency"]
            freq_fit_err = fit_res.params["frequency"].stderr
            old_qb_freq = (
                qubit.parameters.resonance_frequency_ef
                if "f" in self.transition_to_calib
                else qubit.parameters.resonance_frequency_ge
            )
            introduced_detuning = self.experiment_metainfo["detuning"][qubit.uid]
            new_qb_freq = old_qb_freq + introduced_detuning - freq_fit
            self.analysis_results[qubit.uid]["new_parameter_values"].update(
                {"resonance_frequency": new_qb_freq, "T2_star": t2_star}
            )
            self.analysis_results[qubit.uid]["old_parameter_values"].update(
                {"resonance_frequency": old_qb_freq}
            )

            # plot fit
            swpts_fine = np.linspace(swpts_to_fit[0], swpts_to_fit[-1], 501)
            ax.plot(
                swpts_fine * 1e6,
                fit_res.model.func(swpts_fine, **fit_res.best_values),
                "r-",
                zorder=1,
            )
            textstr = (
                f"New qubit frequency: {new_qb_freq / 1e9:.6f} GHz "
                f"$\\pm$ {freq_fit_err / 1e6:.4f} MHz"
            )
            textstr += f"\nOld qubit frequency: {old_qb_freq / 1e9:.6f} GHz"
            textstr += (
                f"\nDiff new-old qubit frequency: "
                f"{(new_qb_freq - old_qb_freq) / 1e6:.6f} MHz"
            )
            textstr += f"\nIntroduced detuning: {introduced_detuning / 1e6:.2f} MHz"
            textstr += (
                f"\nFitted frequency: {freq_fit / 1e6:.6f} "
                f"$\\pm$ {freq_fit_err / 1e6:.4f} MHz"
            )
            textstr += (
                f"\n$T_2^*$: {t2_star * 1e6:.4f} $\\pm$ "
                f"{t2_star_err * 1e6:.4f} $\\mu$s"
            )
            ax.text(0, -0.15, textstr, ha="left", va="top", transform=ax.transAxes)

    def update_qubit_parameters(self):
        for qubit in self.qubits:
            new_qb_pars = self.analysis_results[qubit.uid]["new_parameter_values"]
            if len(new_qb_pars) == 0:
                return

            new_freq = new_qb_pars["resonance_frequency"]
            if "f" in self.transition_to_calib:
                qubit.parameters.resonance_frequency_ef = new_freq
            else:
                qubit.parameters.resonance_frequency_ge = new_freq


class QScale(SingleQubitGateTuneup):
    fallback_experiment_name = "QScale"
    valid_user_parameters = merge_valid_user_parameters(
        [
            dict(
                analysis_metainfo=[
                    "do_fitting",
                    "param_hints",
                ],
            ),
            SingleQubitGateTuneup.valid_user_parameters,
        ]
    )

    def define_experiment(self):
        self.experiment.sections = []
        self.create_acquire_rt_loop()
        self.experiment.add(self.acquire_loop)
        for i, qubit in enumerate(self.qubits):
            swp = self.sweep_parameters_dict[qubit.uid][0]
            # create sweep
            sweep = Sweep(
                uid=f"{qubit.uid}_{self.experiment_name}_sweep", parameters=[swp]
            )
            self.acquire_loop.add(sweep)

            # create pulses sections
            tn = self.transition_to_calib
            X90_pulse = qt_ops.quantum_gate(qubit, f"X90_{tn}")
            X180_pulse = qt_ops.quantum_gate(qubit, f"X180_{tn}")
            Y180_pulse = qt_ops.quantum_gate(qubit, f"Y180_{tn}")
            Ym180_pulse = qt_ops.quantum_gate(qubit, f"mY180_{tn}")
            pulse_ids = ["xx", "xy", "xmy"]
            pulses_2nd = [X180_pulse, Y180_pulse, Ym180_pulse]
            for i, pulse_2nd in enumerate(pulses_2nd):
                id = pulse_ids[i]
                play_after = measure_section if i > 0 else None

                # create preparation pulses sections: ge if calibrating ef
                prep_sections = self.create_qubit_preparation_sections(
                    qubit, play_after=play_after
                )

                # qscale pulses
                play_after_for_exc = prep_sections if len(prep_sections) else play_after
                excitation_section = Section(
                    uid=f"{qubit.uid}_{id}_section",
                    play_after=play_after_for_exc,
                    alignment=SectionAlignment.RIGHT,
                    on_system_grid=True,
                )
                excitation_section.play(
                    signal=self.signal_name(f"drive{self.drive_signal_suffix}", qubit),
                    pulse=X90_pulse,
                    phase=X90_pulse.pulse_parameters["phase"],
                    pulse_parameters={"beta": swp},
                )
                excitation_section.play(
                    signal=self.signal_name(f"drive{self.drive_signal_suffix}", qubit),
                    pulse=pulse_2nd,
                    phase=pulse_2nd.pulse_parameters["phase"],
                    pulse_parameters={"beta": swp},
                )

                # create readout + acquire sections
                measure_section = self.create_measure_acquire_sections(
                    uid=f"measure_{qubit.uid}_{id}",
                    qubit=qubit,
                    play_after=f"{qubit.uid}_{id}_section",
                    handle_suffix=id,
                )

                # Add sections to sweep
                for prep_sec in prep_sections:
                    sweep.add(prep_sec)
                sweep.add(excitation_section)
                sweep.add(measure_section)

            self.add_cal_states_sections(qubit, add_to=self.acquire_loop)

    def analyse_experiment(self):
        ExperimentTemplate.analyse_experiment(self)
        do_pca = self.analysis_metainfo.get("do_pca", False)
        do_fitting = self.analysis_metainfo.get("do_fitting", True)
        for qubit in self.qubits:
            self.analysis_results[qubit.uid]["rotated_data"] = dict()
            if do_fitting:
                self.analysis_results[qubit.uid]["fit_results"] = dict()
            fig, ax = plt.subplots()
            for exp_id in ["xx", "xy", "xmy"]:
                handle = f"{self.experiment_name}_{qubit.uid}_{exp_id}"
                cal_trace_root_handle = f"{self.experiment_name}_{qubit.uid}"
                data_dict = ana_hlp.extract_and_rotate_data_1d(
                    self.results,
                    handle,
                    cal_trace_root_handle,
                    cal_states=self.cal_states,
                    do_pca=do_pca,
                )
                num_cal_traces = data_dict["num_cal_traces"]
                self.analysis_results[qubit.uid]["rotated_data"][
                    f"{exp_id}"
                ] = data_dict
                # plot data
                (line,) = ax.plot(
                    data_dict["sweep_points"],
                    data_dict["data_rotated"],
                    "o",
                    label=exp_id,
                    zorder=2,
                )
                # plot cal traces
                ax.plot(
                    data_dict["sweep_points_cal_traces"],
                    data_dict["data_rotated_cal_traces"],
                    "ok",
                    zorder=2,
                )
                # fit data
                if do_fitting:
                    swpts_to_fit = data_dict["sweep_points"]
                    data_to_fit = data_dict["data_rotated"]

                    if exp_id == "xx":
                        # xx - line at 0.5
                        param_hints = dict(
                            slope=dict(value=0, vary=False),
                            intercept=dict(value=np.mean(data_to_fit)),
                        )
                    elif exp_id == "xy":
                        # fit xy line
                        slope = (data_to_fit[-1] - data_to_fit[0]) / (
                            swpts_to_fit[-1] - swpts_to_fit[0]
                        )
                        intercept = data_to_fit[-1] - slope * swpts_to_fit[-1]
                        param_hints = dict(
                            slope=dict(value=slope), intercept=dict(value=intercept)
                        )
                    elif exp_id == "xmy":
                        # fit xy line
                        slope = (data_to_fit[-1] - data_to_fit[0]) / (
                            swpts_to_fit[-1] - swpts_to_fit[0]
                        )
                        intercept = data_to_fit[-1] - slope * swpts_to_fit[-1]
                        param_hints = dict(
                            slope=dict(value=slope), intercept=dict(value=intercept)
                        )
                    fit_res = ana_hlp.fit_data_lmfit(
                        fit_mods.linear_dependence,
                        swpts_to_fit,
                        data_to_fit,
                        param_hints=param_hints,
                    )
                    self.analysis_results[qubit.uid]["fit_results"][exp_id] = fit_res
                    # plot fit
                    swpts_fine = np.linspace(swpts_to_fit[0], swpts_to_fit[-1], 501)
                    fit_line = fit_res.model.func(swpts_fine, **fit_res.best_values)
                    if not hasattr(fit_line, "__iter__"):  # for xx
                        fit_line = np.repeat(fit_line, len(swpts_fine))
                    ax.plot(swpts_fine, fit_line, c=line.get_color(), zorder=1)

            if do_fitting:
                # calculate optimal qscale
                fit_results = self.analysis_results[qubit.uid]["fit_results"]
                intercept_xy = unc.ufloat(
                    fit_results["xy"].params["intercept"].value,
                    fit_results["xy"].params["intercept"].stderr,
                )
                slope_xy = unc.ufloat(
                    fit_results["xy"].params["slope"].value,
                    fit_results["xy"].params["slope"].stderr,
                )
                intercept_xmy = unc.ufloat(
                    fit_results["xmy"].params["intercept"].value,
                    fit_results["xmy"].params["intercept"].stderr,
                )
                slope_xmy = unc.ufloat(
                    fit_results["xmy"].params["slope"].value,
                    fit_results["xmy"].params["slope"].stderr,
                )
                intercept_diff_mean = intercept_xy - intercept_xmy
                slope_diff_mean = slope_xmy - slope_xy
                qscale = intercept_diff_mean / slope_diff_mean
                self.analysis_results[qubit.uid]["new_parameter_values"].update(
                    {f"{self.transition_to_calib}_beta": qscale.nominal_value}
                )
                old_qscale = (
                    qubit.parameters.drive_parameters_ef["beta"]
                    if "f" in self.transition_to_calib
                    else qubit.parameters.drive_parameters_ge["beta"]
                )
                self.analysis_results[qubit.uid]["old_parameter_values"].update(
                    {f"{self.transition_to_calib}_beta": old_qscale}
                )

                textstr = (
                    f"Quadrature scaling: {qscale.nominal_value:.4f} $\\pm$ "
                    f"{qscale.std_dev:.4f}"
                )
                textstr += f"\nOld value: {old_qscale:.4f}"
                ax.text(0, -0.15, textstr, ha="left", va="top", transform=ax.transAxes)
            ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))
            ax.set_ylabel(
                "Principal Component (a.u)"
                if (num_cal_traces == 0 or do_pca)
                else f"$|{self.cal_states[-1]}\\rangle$-State Population"
            )
            ax.set_xlabel("Quadrature Scaling ($q$)")
            ax.set_title(f"{self.timestamp}_{self.experiment_name}_{qubit.uid}")
            if self.save:
                # Save the figure
                self.save_figure(fig, qubit)
            if self.analysis_metainfo.get("show_figures", False):
                plt.show()
            plt.close(fig)

    def update_qubit_parameters(self):
        for qubit in self.qubits:
            new_qb_pars = self.analysis_results[qubit.uid]["new_parameter_values"]
            if len(new_qb_pars) == 0:
                return

            dr_pars = (
                qubit.parameters.drive_parameters_ef
                if "f" in self.transition_to_calib
                else qubit.parameters.drive_parameters_ge
            )
            dr_pars["beta"] = new_qb_pars[f"{self.transition_to_calib}_beta"]


class T1(SingleQubitGateTuneup):
    fallback_experiment_name = "T1"
    valid_user_parameters = merge_valid_user_parameters(
        [
            dict(
                analysis_metainfo=[
                    "do_fitting",
                    "param_hints",
                ],
            ),
            SingleQubitGateTuneup.valid_user_parameters,
        ]
    )

    def define_experiment(self):
        self.experiment.sections = []
        self.create_acquire_rt_loop()
        self.experiment.add(self.acquire_loop)

        # create joint sweep for all qubits
        sweep = Sweep(
            uid=f"{self.experiment_name}_sweep",
            parameters=[
                self.sweep_parameters_dict[qubit.uid][0] for qubit in self.qubits
            ],
        )
        self.acquire_loop.add(sweep)
        for i, qubit in enumerate(self.qubits):
            # create preparation pulses sections
            prep_sections = self.create_qubit_preparation_sections(qubit)

            # create excitation-pulses section
            excitation_section = Section(
                uid=f"{qubit.uid}_excitation",
                alignment=SectionAlignment.RIGHT,
                on_system_grid=True,
                play_after=prep_sections,
            )
            # add x180 pulse
            x180_pulse = qt_ops.quantum_gate(qubit, f"X180_{self.transition_to_calib}")
            excitation_section.play(
                signal=self.signal_name(f"drive{self.drive_signal_suffix}", qubit),
                pulse=x180_pulse,
            )
            # add delay
            excitation_section.delay(
                signal=self.signal_name(f"drive{self.drive_signal_suffix}", qubit),
                time=self.sweep_parameters_dict[qubit.uid][0],
            )

            # create readout + acquire sections
            measure_section = self.create_measure_acquire_sections(
                qubit=qubit,
                play_after=f"{qubit.uid}_excitation",
            )

            # add sweep and sections to acquire loop rt
            for prep_sec in prep_sections:
                sweep.add(prep_sec)
            sweep.add(excitation_section)
            sweep.add(measure_section)
            self.add_cal_states_sections(qubit, add_to=self.acquire_loop)

    def analyse_experiment_qubit(self, qubit, data_dict, figure, ax):
        # plot data with correct scaling
        ax.plot(
            data_dict["sweep_points_w_cal_traces"] * 1e6,
            data_dict["data_rotated_w_cal_traces"],
            "o",
            zorder=2,
        )
        ax.set_xlabel("Pulse Delay, $\\tau$ ($\\mu$s)")
        if self.analysis_metainfo.get("do_fitting", True):
            swpts_to_fit = data_dict["sweep_points"]
            data_to_fit = data_dict["data_rotated"]
            # fit data
            param_hints = self.analysis_metainfo.get(
                "param_hints",
                {
                    "decay_rate": {"value": 3 * max(swpts_to_fit) / 2},
                    "amplitude": {
                        "value": abs(max(data_to_fit) - min(data_to_fit)) / 2,
                        "min": 0,
                    },
                    "offset": {"value": 0, "vary": False},
                },
            )
            fit_res = ana_hlp.fit_data_lmfit(
                fit_mods.exponential_decay,
                swpts_to_fit,
                data_to_fit,
                param_hints=param_hints,
            )
            self.analysis_results[qubit.uid]["fit_results"] = fit_res

            dec_rt = unc.ufloat(
                fit_res.params["decay_rate"].value, fit_res.params["decay_rate"].stderr
            )
            t1 = 1 / dec_rt
            self.analysis_results[qubit.uid]["new_parameter_values"].update(
                {"T1": t1.nominal_value}
            )

            # plot fit
            swpts_fine = np.linspace(swpts_to_fit[0], swpts_to_fit[-1], 501)
            ax.plot(
                swpts_fine * 1e6,
                fit_res.model.func(swpts_fine, **fit_res.best_values),
                "r-",
                zorder=1,
            )
            textstr = (
                f"$T_1$: {t1.nominal_value * 1e6:.4f} $\\pm$ "
                f"{t1.std_dev * 1e6:.4f} $\\mu$s"
            )
            ax.text(0, -0.15, textstr, ha="left", va="top", transform=ax.transAxes)


class Echo(SingleQubitGateTuneup):
    fallback_experiment_name = "Echo"
    valid_user_parameters = merge_valid_user_parameters(
        [
            dict(
                experiment_metainfo=["detuning"],
                analysis_metainfo=[
                    "do_fitting",
                    "param_hints",
                ],
            ),
            SingleQubitGateTuneup.valid_user_parameters,
        ]
    )

    def define_experiment(self):
        self.experiment.sections = []
        self.create_acquire_rt_loop()
        self.experiment.add(self.acquire_loop)
        # from the delays sweep parameters, create sweep parameters for
        # half the total delay time and for the phase of the second X90 pulse
        detuning = self.experiment_metainfo.get("detuning")
        if detuning is None:
            raise ValueError("Please provide detuning in experiment_metainfo.")
        swp_pars_half_delays = []
        swp_pars_phases = []
        for qubit in self.qubits:
            delays = self.sweep_parameters_dict[qubit.uid][0].values
            pl = (
                qubit.parameters.drive_parameters_ef["length"]
                if "f" in self.transition_to_calib
                else qubit.parameters.drive_parameters_ge["length"]
            )
            swp_pars_half_delays += [
                SweepParameter(uid=f"echo_delays_{qubit.uid}", values=0.5 * delays)
            ]
            swp_pars_phases += [
                SweepParameter(
                    uid=f"echo_phases_{qubit.uid}",
                    values=((delays - delays[0] + pl) * detuning[qubit.uid] * 2 * np.pi)
                    % (2 * np.pi),
                )
            ]

        # create joint sweep for all qubits
        sweep = Sweep(
            uid=f"{self.experiment_name}_sweep",
            parameters=swp_pars_half_delays + swp_pars_phases,
        )
        self.acquire_loop.add(sweep)
        for i, qubit in enumerate(self.qubits):
            # create preparation pulses sections
            prep_sections = self.create_qubit_preparation_sections(qubit)

            # create excitation section
            excitation_section = Section(
                uid=f"{qubit.uid}_excitation",
                alignment=SectionAlignment.RIGHT,
                on_system_grid=True,
                play_after=prep_sections,
            )
            # Echo pulses and delays
            ramsey_drive_pulse = qt_ops.quantum_gate(
                qubit, f"X90_{self.transition_to_calib}"
            )
            echo_drive_pulse = qt_ops.quantum_gate(
                qubit, f"X180_{self.transition_to_calib}"
            )
            excitation_section.play(
                signal=self.signal_name(f"drive{self.drive_signal_suffix}", qubit),
                pulse=ramsey_drive_pulse,
            )
            excitation_section.delay(
                signal=self.signal_name(f"drive{self.drive_signal_suffix}", qubit),
                time=swp_pars_half_delays[i],
            )
            excitation_section.play(
                signal=self.signal_name(f"drive{self.drive_signal_suffix}", qubit),
                pulse=echo_drive_pulse,
            )
            excitation_section.delay(
                signal=self.signal_name(f"drive{self.drive_signal_suffix}", qubit),
                time=swp_pars_half_delays[i],
            )
            excitation_section.play(
                signal=self.signal_name(f"drive{self.drive_signal_suffix}", qubit),
                pulse=ramsey_drive_pulse,
                phase=swp_pars_phases[i],
            )

            # create readout + acquire sections
            measure_section = self.create_measure_acquire_sections(
                qubit=qubit,
                play_after=f"{qubit.uid}_excitation",
            )

            # add sweep and sections to acquire loop rt
            for prep_sec in prep_sections:
                sweep.add(prep_sec)
            sweep.add(excitation_section)
            sweep.add(measure_section)
            self.add_cal_states_sections(qubit, add_to=self.acquire_loop)

    def analyse_experiment_qubit(self, qubit, data_dict, figure, ax):
        delays_offset = (
            qubit.parameters.drive_parameters_ef["length"]
            if "f" in self.transition_to_calib
            else qubit.parameters.drive_parameters_ge["length"]
        )
        # plot data with correct scaling
        ax.plot(
            (data_dict["sweep_points_w_cal_traces"] + delays_offset) * 1e6,
            data_dict["data_rotated_w_cal_traces"],
            "o",
            zorder=2,
        )
        ax.set_xlabel("Pulse Separation, $\\tau$ ($\\mu$s)")

        if self.analysis_metainfo.get("do_fitting", True):
            swpts_to_fit = data_dict["sweep_points"] + delays_offset
            data_to_fit = data_dict["data_rotated"]
            # fit data
            freqs_guess, phase_guess = ana_hlp.find_oscillation_frequency_and_phase(
                data_to_fit, swpts_to_fit
            )
            param_hints = self.analysis_metainfo.get(
                "param_hints",
                {
                    "frequency": {"value": freqs_guess},
                    "phase": {"value": phase_guess},
                    "decay_time": {"value": 2 / 3 * max(swpts_to_fit), "min": 0},
                    "amplitude": {"value": 0.5, "vary": False},
                    "oscillation_offset": {"value": 0, "vary": "f" in self.cal_states},
                    "exponential_offset": {"value": np.mean(data_to_fit)},
                    "decay_exponent": {"value": 1, "vary": False},
                },
            )
            fit_res = ana_hlp.fit_data_lmfit(
                fit_mods.oscillatory_decay_flexible,
                swpts_to_fit,
                data_to_fit,
                param_hints=param_hints,
            )
            self.analysis_results[qubit.uid]["fit_results"] = fit_res

            t2 = fit_res.best_values["decay_time"]
            t2_err = fit_res.params["decay_time"].stderr
            self.analysis_results[qubit.uid]["new_parameter_values"].update({"T2": t2})

            # plot fit
            swpts_fine = np.linspace(swpts_to_fit[0], swpts_to_fit[-1], 501)
            ax.plot(
                swpts_fine * 1e6,
                fit_res.model.func(swpts_fine, **fit_res.best_values),
                "r-",
                zorder=1,
            )
            textstr = f"$T_2$: {t2 * 1e6:.4f} $\\pm$ " f"{t2_err * 1e6:.4f} $\\mu$s"
            ax.text(0, -0.15, textstr, ha="left", va="top", transform=ax.transAxes)


class RamseyParking(Ramsey):
    fallback_experiment_name = "RamseyParking"
    valid_user_parameters = merge_valid_user_parameters(
        [
            dict(
                experiment_metainfo=["neartime_callback_function"],
                analysis_metainfo=["param_hints_parking_fit"],
            ),
            Ramsey.valid_user_parameters,
        ]
    )

    def __init__(self, *args, **kwargs):
        apply_exit_condition = kwargs.get("apply_exit_condition", True)
        kwargs["apply_exit_condition"] = apply_exit_condition
        super().__init__(*args, **kwargs)

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
                "sweep in experiment_metainfo['neartime_callback_function']."
            )
        # all near-time callback functions have the format
        # func(session, sweep_param_value, qubit)
        nt_sweep.call(ntsf, voltage=nt_sweep_par, qubit=qubit)
        # self.create_acquire_rt_loop()
        nt_sweep.add(self.acquire_loop)

    def analyse_experiment(self):
        ExperimentTemplate.analyse_experiment(self)
        for qubit in self.qubits:
            # extract data
            handle = f"{self.experiment_name}_{qubit.uid}"
            do_pca = self.analysis_metainfo.get("do_pca", False)
            data_dict = ana_hlp.extract_and_rotate_data_2d(
                self.results, handle, cal_states=self.cal_states, do_pca=do_pca
            )
            num_cal_traces = data_dict["num_cal_traces"]

            if self.analysis_metainfo.get("do_fitting", True):
                voltages = data_dict["sweep_points_nt"]
                ramsey_analysis_results = {}
                # run Ramsey analysis
                data_to_fit_2d = data_dict["data_rotated"]
                data_rotated_w_cal_tr_2d = data_dict["data_rotated_w_cal_traces"]
                for i in range(data_to_fit_2d.shape[0]):
                    data_to_fit = data_to_fit_2d[i, :]
                    data_dict_tmp = deepcopy(data_dict)
                    data_dict_tmp["data_rotated"] = data_to_fit
                    data_dict_tmp[
                        "data_rotated_w_cal_traces"
                    ] = data_rotated_w_cal_tr_2d[i, :]
                    self.analysis_results[qubit.uid]["rotated_data"] = data_dict

                    fig, ax = plt.subplots()
                    ax.set_xlabel("Pulse Separation, $\\tau$ ($\\mu$s)")
                    ax.set_ylabel(
                        "Principal Component (a.u)"
                        if (num_cal_traces == 0 or do_pca)
                        else f"$|{self.cal_states[-1]}\\rangle$-State Population"
                    )
                    ax.set_title(f"{self.timestamp}_{handle}")
                    # run ramsey analysis
                    self.analyse_experiment_qubit(qubit, data_dict_tmp, fig, ax)
                    if self.save:
                        # Save the figure
                        fig_name = (
                            f"{self.timestamp}_Ramsey"
                            f"_{qubit.uid}_{voltages[i]:.3f}V"
                        )
                        self.save_figure(fig, qubit, fig_name)
                    plt.close(fig)
                    ramsey_analysis_results[f"Ramsey_{i}"] = deepcopy(
                        self.analysis_results[qubit.uid]
                    )

                self.analysis_results[qubit.uid].update(ramsey_analysis_results)
                # fit qubit frequencies vs voltage
                qubit_frequencies = np.array(
                    [
                        self.analysis_results[qubit.uid][f"Ramsey_{i}"][
                            "new_parameter_values"
                        ]["resonance_frequency"]
                        for i in range(len(voltages))
                    ]
                )
                self.analysis_results[qubit.uid][
                    "qubit_frequencies"
                ] = qubit_frequencies

                # figure out whether voltages vs freqs is convex or concave
                take_extremum_fit, scf = (
                    (np.argmax, 1)
                    if (ana_hlp.is_data_convex(voltages, qubit_frequencies))
                    else (np.argmin, -1)
                )
                # optimal parking parameters at the extremum of
                # voltages vs frequencies
                f0 = qubit_frequencies[take_extremum_fit(qubit_frequencies)]
                V0 = voltages[take_extremum_fit(qubit_frequencies)]
                param_hints = self.analysis_metainfo.get(
                    "param_hints",
                    {
                        "voltage_sweet_spot": {"value": V0},
                        "frequency_sweet_spot": {"value": f0},
                        "frequency_voltage_scaling": {
                            "value": scf
                            * (max(qubit_frequencies) - min(qubit_frequencies))
                        },
                    },
                )
                fit_res = ana_hlp.fit_data_lmfit(
                    fit_mods.transmon_voltage_dependence_quadratic,
                    voltages,
                    qubit_frequencies,
                    param_hints=param_hints,
                )
                f0 = fit_res.best_values["frequency_sweet_spot"]
                f0err = fit_res.params["frequency_sweet_spot"].stderr
                V0 = fit_res.best_values["voltage_sweet_spot"]
                V0err = fit_res.params["voltage_sweet_spot"].stderr
                self.analysis_results[qubit.uid]["fit_results"] = fit_res
                self.analysis_results[qubit.uid]["new_parameter_values"] = {
                    f"resonance_frequency_{self.transition_to_calib}": f0,
                    "dc_voltage_parking": V0,
                }
                V0_old = qubit.parameters.dc_voltage_parking
                f0_old = qubit.parameters.resonance_frequency_ge
                self.analysis_results[qubit.uid]["old_parameter_values"] = {
                    f"resonance_frequency_{self.transition_to_calib}": f0_old,
                    "dc_voltage_parking": V0_old,
                }
                # plot data + fit
                fig, ax = plt.subplots()
                ax.set_xlabel(self.results.get_axis_name(handle)[0])
                ax.set_ylabel("Qubit Frequency, $f_{\\mathrm{QB}}$ (GHz)")
                ax.set_title(f"{self.timestamp}_{handle}")
                ax.plot(voltages, qubit_frequencies / 1e9, "o", zorder=2)
                # plot fit
                voltages_fine = np.linspace(voltages[0], voltages[-1], 501)
                ax.plot(
                    voltages_fine,
                    fit_res.model.func(voltages_fine, **fit_res.best_values) / 1e9,
                    "r-",
                    zorder=1,
                )
                if voltages[0] <= V0 <= voltages[-1]:
                    ax.plot(
                        V0,
                        f0 / 1e9,
                        "sk",
                        markersize=plt.rcParams["lines.markersize"] + 1,
                    )
                textstr = f"Parking voltage: {V0:.4f} $\\pm$ {V0err:.4f} V (Old value: {V0_old:.4f} V)"
                textstr += (
                    f"\nParking frequency: {f0 / 1e9:.6f} $\\pm$ {f0err / 1e9:.6f} GHz "
                    f"(Old value: {f0_old / 1e9:.6f} GHz)"
                )
                ax.text(0, -0.15, textstr, ha="left", va="top", transform=ax.transAxes)

                # save figures and results
                if self.save:
                    # Save the figure
                    self.save_figure(fig, qubit)
                if self.analysis_metainfo.get("show_figures", False):
                    plt.show()
                plt.close(fig)

    def execute_exit_condition(self):
        # set the dc voltage source to the original voltage value stored in the
        # qubit parameters
        nt_cb_func = self.experiment_metainfo.get("neartime_callback_function", None)
        if nt_cb_func is None:
            raise ValueError(
                "Please provide the neartime callback function for setting the "
                "dc voltage in experiment_metainfo['neartime_callback_function']."
            )
        for qubit in self.qubits:
            log.info(
                f"Setting DC voltage source slot "
                f"{qubit.parameters.dc_slot} ({qubit.uid}) back to the "
                f"original value of {qubit.parameters.dc_voltage_parking:.4f}."
            )
            nt_cb_func(self.session, qubit.parameters.dc_voltage_parking, qubit)

    def update_qubit_parameters(self):
        for qubit in self.qubits:
            new_qb_pars = self.analysis_results[qubit.uid]["new_parameter_values"]
            if len(new_qb_pars) == 0:
                return

            new_freq = new_qb_pars[f"resonance_frequency_{self.transition_to_calib}"]
            if "f" in self.transition_to_calib:
                qubit.parameters.resonance_frequency_ef = new_freq
            else:
                qubit.parameters.resonance_frequency_ge = new_freq
            qubit.parameters.dc_voltage_parking = new_qb_pars["dc_voltage_parking"]

    def update_entire_setup(self):
        super().update_entire_setup()
        # set the dc voltage source to the new voltage value
        nt_cb_func = self.experiment_metainfo.get("neartime_callback_function", None)
        if nt_cb_func is None:
            raise ValueError(
                "Please provide the neartime callback function for setting the "
                "dc voltage in experiment_metainfo['neartime_callback_function']."
            )
        for qubit in self.qubits:
            log.info(
                f"Updating DC voltage source slot "
                f"{qubit.parameters.dc_slot} ({qubit.uid}) to the "
                f"new value of {qubit.parameters.dc_voltage_parking:.4f}."
            )
            nt_cb_func(self.session, qubit.parameters.dc_voltage_parking, qubit)
