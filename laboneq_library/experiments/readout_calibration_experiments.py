import numpy as np
from copy import deepcopy
import uncertainties as unc
from itertools import combinations
import matplotlib.pyplot as plt
from ruamel.yaml import YAML

ryaml = YAML()

import logging

log = logging.getLogger(__name__)

from . import quantum_operations as qt_ops
from laboneq.analysis import fitting as fit_mods
from laboneq.analysis import calculate_integration_kernels
from laboneq.simple import *  # noqa: F403
from laboneq_library.analysis import analysis_helpers as ana_hlp
from laboneq_library.experiments.experiment_library import (
    ExperimentTemplate,
    merge_valid_user_parameters,
)

#### RAW traces Experiments ####


class OptimalIntegrationKernels(ExperimentTemplate):
    fallback_experiment_name = "OptimalIntegrationKernels"
    valid_user_parameters = merge_valid_user_parameters(
        [
            dict(
                analysis_metainfo=[
                    "show_figures",
                ],
            ),
            ExperimentTemplate.valid_user_parameters,
        ]
    )

    def __init__(self, *args, preparation_states=("g", "e"), **kwargs):
        self.preparation_states = preparation_states
        acquisition_metainfo_user = kwargs.pop("acquisition_metainfo", dict())
        acquisition_metainfo = dict(acquisition_type=AcquisitionType.RAW)
        acquisition_metainfo.update(acquisition_metainfo_user)
        kwargs["acquisition_metainfo"] = acquisition_metainfo

        # Add suffix to experiment name
        experiment_name = kwargs.get("experiment_name", self.fallback_experiment_name)
        experiment_name += f"_{''.join(self.preparation_states)}"
        kwargs["experiment_name"] = experiment_name

        run = kwargs.pop("run", False)  # instantiate base without running exp
        kwargs["run"] = False
        save = kwargs.pop("save", True)  # instantiate base without saving exp
        kwargs["save"] = save
        super().__init__(*args, **kwargs)

        # create experiment for each prep state
        self.experiments = {}
        kwargs_exp = deepcopy(kwargs)
        kwargs_exp.pop("signals", None)
        if "g" in self.preparation_states:
            self.experiments["g"] = ExperimentTemplate(*args, **kwargs_exp)
            self.experiments["g"].experiment_name = f"{self.experiment_name}_g"
        if "e" in self.preparation_states:
            self.experiments["e"] = ExperimentTemplate(
                *args, signals=["measure", "acquire", "drive"], **kwargs_exp
            )
            self.experiments["e"].experiment_name = f"{self.experiment_name}_e"
        if "f" in self.preparation_states:
            self.experiments["f"] = ExperimentTemplate(
                *args, signals=["measure", "acquire", "drive", "drive_ef"], **kwargs_exp
            )
            self.experiments["f"].experiment_name = f"{self.experiment_name}_f"

        self.save = save
        self.run = run
        if self.run:
            self.autorun()

    def define_experiment(self):
        # self.exp_g.create_acquire_rt_loop()
        # self.exp_g.experiment.add(self.exp_g.acquire_loop)
        # self.exp_e.create_acquire_rt_loop()
        # self.exp_e.experiment.add(self.exp_e.acquire_loop)

        for prep_state, exp in self.experiments.items():
            exp.create_acquire_rt_loop()
            exp.experiment.add(exp.acquire_loop)

            for qubit in self.qubits:
                measure_play_after = None
                excitation_sections = []
                if prep_state in ["e", "f"]:
                    # create ge-preparation section
                    excitation_ge_section = Section(
                        uid=f"{qubit.uid}_ge_excitation",
                        alignment=SectionAlignment.RIGHT,
                        on_system_grid="f" in self.preparation_states,
                    )
                    # ge-preparation drive pulse
                    drive_pulse_ge = qt_ops.quantum_gate(qubit, f"X180_ge")
                    excitation_ge_section.play(
                        signal=self.signal_name(f"drive", qubit),
                        pulse=drive_pulse_ge,
                    )
                    measure_play_after = f"{qubit.uid}_ge_excitation"
                    excitation_sections += [excitation_ge_section]
                    exp.acquire_loop.add(excitation_ge_section)
                if prep_state == "f":
                    # create ef-preparation section
                    excitation_ef_section = Section(
                        uid=f"{qubit.uid}_ef_excitation",
                        alignment=SectionAlignment.RIGHT,
                        on_system_grid=True,
                        play_after=f"{qubit.uid}_ge_excitation",
                    )
                    # ef-preparation drive pulse
                    drive_pulse_ef = qt_ops.quantum_gate(qubit, f"X180_ef")
                    excitation_ef_section.play(
                        signal=self.signal_name(f"drive_ef", qubit),
                        pulse=drive_pulse_ef,
                    )
                    measure_play_after = f"{qubit.uid}_ef_excitation"
                    excitation_sections += [excitation_ef_section]
                    exp.acquire_loop.add(excitation_ef_section)

                measure_sections = exp.create_measure_acquire_sections(
                    qubit=qubit, integration_kernel=None, play_after=measure_play_after
                )
                exp.acquire_loop.add(measure_sections)

    def configure_experiment(self):
        for exp in self.experiments.values():
            exp.configure_experiment()
            cal = Calibration()
            for qubit in self.qubits:
                cal[f"acquire_{qubit.uid}"] = SignalCalibration(
                    oscillator=None,
                    port_delay=240e-9,
                )
            exp.experiment.set_calibration(cal)

    def compile_experiment(self):
        # self.exp_g.compile_experiment()
        # self.exp_e.compile_experiment()
        for exp in self.experiments.values():
            exp.compile_experiment()

    def run_experiment(self):
        # self.exp_g.run_experiment()
        # self.exp_e.run_experiment()
        for exp in self.experiments.values():
            exp.run_experiment()
            # save experiments to the same directory, under the same timestamp
            exp.timestamp = self.timestamp
            exp.save_directory = self.save_directory

    def save_results(self):
        for state, exp in self.experiments.items():
            exp.save_results(filename_suffix=state)

    def analyse_experiment(self):
        super().analyse_experiment()
        for qubit in self.qubits:
            # plot traces and kernel
            fig_size = plt.rcParams["figure.figsize"]
            fig, axs = plt.subplots(
                nrows=len(self.preparation_states)
                + 1
                + ("f" in self.preparation_states),
                sharex=True,
                figsize=(fig_size[0], fig_size[1] * 2),
            )

            raw_traces = []
            for i, ps in enumerate(self.preparation_states):
                handle = f"{self.experiment_name}_{ps}_{qubit.uid}"
                trace = self.experiments[ps].results.get_data(handle)
                raw_traces += [trace[: (len(trace) // 16) * 16]]
                axs[i].plot(np.real(raw_traces[-1]), label=f"{ps}: I")
                axs[i].plot(np.imag(raw_traces[-1]), label=f"{ps}: Q")
                axs[i].set_ylabel("Voltage, $V$ (a.u.)")
                axs[i].legend(frameon=False)
            self.analysis_results[qubit.uid]["raw_traces"] = {
                ps: raw_traces[i] for i, ps in enumerate(self.preparation_states)
            }

            kernels = calculate_integration_kernels(raw_traces)
            self.analysis_results[qubit.uid]["new_parameter_values"].update(
                {"integration_kernels": kernels}
            )
            for i, krn in enumerate(kernels):
                ax = axs[len(self.preparation_states) + i]
                krn_vals = krn.samples
                ax.plot(np.real(krn_vals), label=f"w{i+1}: I")
                ax.plot(np.imag(krn_vals), label=f"w{i+1}: Q")
                ax.set_ylabel("Voltage, $V$ (a.u.)")
                ax.legend(frameon=False)
            axs[-1].set_xlabel("Samples, $N$")
            axs[0].set_title(f"{self.timestamp}_{self.experiment_name}_{qubit.uid}")
            fig.align_ylabels()
            fig.subplots_adjust(hspace=0.05)

            # save figures
            if self.save:
                # Save the figures
                self.save_figure(fig, qubit)
            if self.analysis_metainfo.get("show_figures", False):
                plt.show()
            plt.close(fig)

    def update_qubit_parameters(self):
        for qubit in self.qubits:
            new_qb_pars = self.analysis_results[qubit.uid]["new_parameter_values"]
            if len(new_qb_pars) == 0:
                return
            qubit.parameters.readout_integration_kernels = new_qb_pars[
                "integration_kernels"
            ]


###################################
#### Spectroscopy Experiments  ####
###################################


class ResonatorSpectroscopy(ExperimentTemplate):
    fallback_experiment_name = "ResonatorSpectroscopy"
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
        kwargs["signals"] = kwargs.pop("signals", ["measure", "acquire"])

        acquisition_metainfo_user = kwargs.pop("acquisition_metainfo", dict())
        acquisition_metainfo = dict(acquisition_type=AcquisitionType.SPECTROSCOPY)
        acquisition_metainfo.update(acquisition_metainfo_user)
        kwargs["acquisition_metainfo"] = acquisition_metainfo

        experiment_metainfo = kwargs.get("experiment_metainfo", dict())
        self.nt_swp_par = experiment_metainfo.get("neartime_sweep_parameter", None)
        self.pulsed = experiment_metainfo.get("pulsed", False)
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
                    values=freq_swp.values - qubit.parameters.readout_lo_frequency,
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
        for qubit in self.qubits:
            ro_pulse_amp = qubit.parameters.readout_amplitude
            qb_sweep_pars = self.sweep_parameters_dict[qubit.uid]
            if len(qb_sweep_pars) > 1:
                nt_sweep_par = qb_sweep_pars[1]
                nt_sweep = Sweep(
                    uid=f"neartime_{self.nt_swp_par}_sweep_{qubit.uid}",
                    parameters=[nt_sweep_par],
                )
                self.experiment.add(nt_sweep)
                if self.nt_swp_par == "voltage":
                    ntsf = self.experiment_metainfo.get(
                        "neartime_callback_function", None
                    )
                    if ntsf is None:
                        raise ValueError(
                            "Please provide the neartime callback function for "
                            "the voltage sweep in "
                            "experiment_metainfo['neartime_callback_function']."
                        )
                    # all near-time callback functions have the format
                    # func(session, sweep_param_value, qubit)
                    nt_sweep.call(ntsf, voltage=nt_sweep_par, qubit=qubit)
                elif self.nt_swp_par == "amplitude":
                    ro_pulse_amp = 1
                # add real-time loop to nt_sweep
                nt_sweep.add(self.acquire_loop)
            else:
                self.experiment.add(self.acquire_loop)

            inner_freq_sweep = qb_sweep_pars[0]
            sweep_inner = Sweep(
                uid=f"resonator_frequency_inner_{qubit.uid}",
                parameters=[inner_freq_sweep],
            )
            measure_acquire_section = Section(uid=f"measure_acquire_{qubit.uid}")
            if self.pulsed:
                ro_pulse = pulse_library.const(
                    length=qubit.parameters.readout_pulse_length, amplitude=ro_pulse_amp
                )
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
                    reset_delay=qubit.parameters.reset_delay_length,
                )
                sweep_inner.add(measure_acquire_section)
            else:
                measure_acquire_section.measure(
                    measure_signal=None,
                    handle=f"{self.experiment_name}_{qubit.uid}",
                    acquire_signal=self.signal_name("acquire", qubit),
                    integration_length=qubit.parameters.readout_integration_length,
                    reset_delay=qubit.parameters.reset_delay_length,
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
        cal = Calibration()
        for qubit in self.qubits:
            qb_sweep = self.sweep_parameters_dict[qubit.uid]
            local_oscillator = None
            ro_amplitude = None
            if self.nt_swp_par == "amplitude":  # and not self.pulsed:
                ro_amplitude = qb_sweep[1]
            elif self.nt_swp_par == "frequency":
                local_oscillator = Oscillator(frequency=qb_sweep[1])

            freq_swp = qb_sweep[0]
            meas_sig_calib_kwargs = {}
            if local_oscillator is not None:
                meas_sig_calib_kwargs["local_oscillator"] = local_oscillator
                cal[self.signal_name("acquire", qubit)] = SignalCalibration(
                    local_oscillator=local_oscillator,
                )
            if ro_amplitude is not None:
                meas_sig_calib_kwargs["amplitude"] = ro_amplitude
            cal[self.signal_name("measure", qubit)] = SignalCalibration(
                oscillator=Oscillator(
                    frequency=freq_swp, modulation_type=ModulationType.HARDWARE
                ),
                **meas_sig_calib_kwargs,
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

            # get frequency filter of qubit
            ff_qb = freq_filter.get(qubit.uid, None)
            # decide whether to extract peaks or dips for qubit
            fp_qb = find_peaks.get(qubit.uid, False)
            take_extremum = np.argmax if fp_qb else np.argmin
            # extract data
            handle = f"{self.experiment_name}_{qubit.uid}"
            data_mag = abs(self.results.get_data(handle))
            res_axis = self.results.get_axis(handle)
            if self.nt_swp_par is None or self.nt_swp_par == "frequency":
                data_mag = np.array([data for data in data_mag]).flatten()
                if len(res_axis) > 1:
                    outer = self.results.get_axis(handle)[0]
                    inner = self.results.get_axis(handle)[1]
                    freqs = np.array([out + inner for out in outer]).flatten()
                else:
                    freqs = (
                        self.results.get_axis(handle)[0]
                        + qubit.parameters.readout_lo_frequency
                    )

                # plot data
                fig, ax = plt.subplots()
                ax.plot(freqs / 1e9, data_mag)
                ax.set_xlabel("Readout Frequency, $f_{\\mathrm{RO}}$ (GHz)")
                ax.set_ylabel("Signal Magnitude, $|S_{21}|$ (a.u.)")

                if self.analysis_metainfo.get("do_fitting", True):
                    data_to_search = (
                        data_mag if ff_qb is None else data_mag[ff_qb(freqs)]
                    )
                    freqs_to_search = freqs if ff_qb is None else freqs[ff_qb(freqs)]
                    f0 = freqs_to_search[take_extremum(data_to_search)]
                    d0 = data_to_search[take_extremum(data_to_search)]
                    new_parameter_values["readout_resonator_frequency"] = f0
                    ax.plot(f0 / 1e9, d0, "ro")
                    textstr = (
                        f"Extracted readout-resonator frequency: {f0 / 1e9:.4f} GHz"
                    )
                    textstr += (
                        f"\nCurrent readout-resonator frequency: "
                        f"{qubit.parameters.readout_resonator_frequency / 1e9:.4f} GHz"
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
                    + qubit.parameters.readout_lo_frequency
                )
                data_mag = abs(self.results.get_data(handle))

                X, Y = np.meshgrid(freqs / 1e9, nt_sweep_par_vals)
                fig, ax = plt.subplots(constrained_layout=True)

                CS = ax.contourf(X, Y, data_mag, levels=100, cmap="magma")
                ax.set_title(f"{handle}")
                ax.set_xlabel("Readout Frequency, $f_{\\mathrm{RO}}$ (GHz)")
                ax.set_ylabel(nt_sweep_par_name)
                cbar = fig.colorbar(CS)
                cbar.set_label("Signal Magnitude, $|S_{21}|$ (a.u.)")

                if self.nt_swp_par == "voltage":
                    # 1D plot of the qubit frequency vs voltage
                    if ff_qb is None:
                        freqs_dips = freqs[take_extremum(data_mag, axis=1)]
                    else:
                        mask = ff_qb(freqs)
                        freqs_dips = freqs[mask][
                            take_extremum(data_mag[:, mask], axis=1)
                        ]
                    # plot
                    ax.plot(freqs_dips / 1e9, nt_sweep_par_vals, "ow")
                    # figure out whether voltages vs freqs is convex or concave
                    take_extremum_fit, scf = (
                        (np.argmax, 1)
                        if (ana_hlp.is_data_convex(nt_sweep_par_vals, freqs_dips))
                        else (np.argmin, -1)
                    )
                    # optimal parking parameters at the extremum of
                    # voltages vs frequencies
                    f0 = freqs_dips[take_extremum_fit(freqs_dips)]
                    V0 = nt_sweep_par_vals[take_extremum_fit(freqs_dips)]
                    new_parameter_values.update(
                        {"readout_resonator_frequency": f0, "dc_voltage_parking": V0}
                    )

                    if self.analysis_metainfo.get("do_fitting", True):
                        # fit frequency vs voltage and take the optimal parking
                        # parameters from fit
                        data_to_fit = freqs_dips
                        swpts_to_fit = nt_sweep_par_vals
                        (
                            freqs_guess,
                            phase_guess,
                        ) = ana_hlp.find_oscillation_frequency_and_phase(
                            data_to_fit, swpts_to_fit
                        )
                        param_hints = self.analysis_metainfo.get(
                            "param_hints",
                            {
                                "frequency": {
                                    "value": 2 * np.pi * freqs_guess,
                                    "min": 0,
                                },
                                "phase": {"value": phase_guess},
                                "amplitude": {
                                    "value": abs(max(data_to_fit) - min(data_to_fit))
                                    / 2,
                                    "min": 0,
                                },
                                "offset": {"value": np.mean(data_to_fit)},
                            },
                        )
                        fit_res = ana_hlp.fit_data_lmfit(
                            fit_mods.oscillatory,
                            swpts_to_fit,
                            data_to_fit,
                            param_hints=param_hints,
                        )
                        self.analysis_results[qubit.uid]["fit_results"] = fit_res

                        # extract USS and LSS voltages and frequencies
                        freq_fit = unc.ufloat(
                            fit_res.params["frequency"].value,
                            fit_res.params["frequency"].stderr,
                        )
                        phase_fit = unc.ufloat(
                            fit_res.params["phase"].value,
                            fit_res.params["phase"].stderr,
                        )
                        (
                            voltages_uss,
                            voltages_lss,
                            _,
                            _,
                        ) = ana_hlp.get_pi_pi2_xvalues_on_cos(
                            swpts_to_fit, freq_fit, phase_fit
                        )
                        v_uss_values = np.array(
                            [vuss.nominal_value for vuss in voltages_uss]
                        )
                        v_lss_values = np.array(
                            [vlss.nominal_value for vlss in voltages_lss]
                        )
                        freqs_uss = fit_res.model.func(
                            v_uss_values, **fit_res.best_values
                        )
                        freqs_lss = fit_res.model.func(
                            v_lss_values, **fit_res.best_values
                        )

                        # plot fit
                        swpts_fine = np.linspace(swpts_to_fit[0], swpts_to_fit[-1], 501)
                        ax.plot(
                            fit_res.model.func(swpts_fine, **fit_res.best_values) / 1e9,
                            swpts_fine,
                            "w-",
                        )
                        (line_uss,) = ax.plot(freqs_uss / 1e9, v_uss_values, "bo")
                        (line_lss,) = ax.plot(freqs_lss / 1e9, v_lss_values, "go")

                        # extract parking values, show them on plot and save
                        # them in self.new_qubit_parameters
                        new_parameter_values.update(
                            {
                                "readout_resonator_frequency": {},
                                "dc_voltage_parking": {},
                            }
                        )
                        if len(v_uss_values) > 0:
                            uss_idx = np.argsort(abs(v_uss_values))[0]
                            v_uss, f_uss = voltages_uss[uss_idx], freqs_uss[uss_idx]
                            textstr = (
                                f"Smallest USS voltage:\n"
                                + f"{v_uss.nominal_value:.4f} V $\\pm$ {v_uss.std_dev:.4f} V"
                            )
                            textstr += f"\nParking frequency:\n{f_uss / 1e9:.4f} GHz"
                            ax.text(
                                1,
                                -0.15,
                                textstr,
                                ha="right",
                                va="top",
                                c=line_uss.get_c(),
                                transform=ax.transAxes,
                            )
                            new_parameter_values["readout_resonator_frequency"][
                                "uss"
                            ] = f_uss
                            new_parameter_values["dc_voltage_parking"][
                                "uss"
                            ] = v_uss.nominal_value
                        if len(v_lss_values) > 0:
                            lss_idx = np.argsort(abs(v_lss_values))[0]
                            v_lss, f_lss = voltages_lss[lss_idx], freqs_lss[lss_idx]
                            textstr = (
                                f"Smallest LSS voltage:\n"
                                + f"{v_lss.nominal_value:.4f} V $\\pm$ {v_lss.std_dev:.4f} V"
                            )
                            textstr += f"\nParking frequency:\n{f_lss / 1e9:.4f} GHz"
                            ax.text(
                                0,
                                -0.15,
                                textstr,
                                ha="left",
                                va="top",
                                c=line_lss.get_c(),
                                transform=ax.transAxes,
                            )
                            new_parameter_values["readout_resonator_frequency"][
                                "lss"
                            ] = f_lss
                            new_parameter_values["dc_voltage_parking"][
                                "lss"
                            ] = v_lss.nominal_value

            ax.set_title(f"{self.timestamp}_{handle}")
            # save figures and results
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

            new_rr_freq = new_qb_pars["readout_resonator_frequency"]
            if isinstance(new_rr_freq, dict):
                # both uss and lss found
                ss_to_update = self.analysis_metainfo.get("sweet_spot_to_update", None)
                if ss_to_update is None:
                    raise ValueError(
                        "Both upper and lower sweep spots were found. "
                        "Unclear which one to set. Please update "
                        'specify "sweet_spot_to_update" in '
                        "analysis_metainfo."
                    )
                qubit.parameters.readout_resonator_frequency = new_rr_freq[ss_to_update]
                qubit.parameters.dc_voltage_parking = new_qb_pars["dc_voltage_parking"][
                    ss_to_update
                ]
            else:
                qubit.parameters.readout_resonator_frequency = new_rr_freq
                if "dc_voltage_parking" in new_qb_pars:
                    qubit.parameters.dc_voltage_parking = new_qb_pars[
                        "dc_voltage_parking"
                    ]


class DispersiveShift(ResonatorSpectroscopy):
    fallback_experiment_name = "DispersiveShift"
    valid_user_parameters = merge_valid_user_parameters(
        [
            dict(
                analysis_metainfo=[
                    "show_figures",
                ],
            ),
            ResonatorSpectroscopy.valid_user_parameters,
        ]
    )

    def __init__(self, *args, preparation_states=("g", "e"), **kwargs):
        self.preparation_states = preparation_states
        experiment_metainfo = kwargs.get("experiment_metainfo", dict())
        experiment_metainfo["pulsed"] = True
        kwargs["experiment_metainfo"] = experiment_metainfo
        run = kwargs.pop("run", False)  # instantiate base without running exp
        kwargs["run"] = False
        save = kwargs.pop("save", True)  # instantiate base without saving exp
        kwargs["save"] = save

        super().__init__(*args, **kwargs)
        # remove the Pulsed suffix from the experiment name
        exp_name_split = self.experiment_name.split("_")
        exp_name_split.remove("Pulsed")
        self.experiment_name = "_".join(exp_name_split)
        self.experiment_name += f"_{''.join(preparation_states)}"
        self.create_experiment_label()
        self.generate_timestamp_save_directory()

        # create ResonatorSpectroscopy experiments for each prep state
        self.experiments = dict()
        kwargs_exp = deepcopy(kwargs)
        kwargs_exp.pop("signals", None)
        if "g" in self.preparation_states:
            self.experiments["g"] = ResonatorSpectroscopy(*args, **kwargs_exp)
            self.experiments["g"].experiment_name = f"{self.experiment_name}_g"
        if "e" in self.preparation_states:
            self.experiments["e"] = ResonatorSpectroscopy(
                *args, signals=["measure", "acquire", "drive"], **kwargs_exp
            )
            self.experiments["e"].experiment_name = f"{self.experiment_name}_e"
        if "f" in self.preparation_states:
            self.experiments["f"] = ResonatorSpectroscopy(
                *args, signals=["measure", "acquire", "drive", "drive_ef"], **kwargs_exp
            )
            self.experiments["f"].experiment_name = f"{self.experiment_name}_f"

        self.save = save
        self.run = run
        if self.run:
            self.autorun()

    def define_experiment(self):
        for exp in self.experiments.values():
            exp.define_experiment()

        for prep_state in ["e", "f"]:
            if prep_state not in self.experiments:
                continue
            exp = self.experiments[prep_state].experiment
            acq_rt_loop = exp.sections[0]
            for qubit in self.qubits:
                freq_sweep_idx, freq_sweep = [
                    (i, chld)
                    for i, chld in enumerate(acq_rt_loop.children)
                    if qubit.uid in chld.uid
                ][0]

                excitation_sections = []
                # create ge-preparation section
                excitation_ge_section = Section(
                    uid=f"{qubit.uid}_ge_excitation",
                    alignment=SectionAlignment.RIGHT,
                    on_system_grid="f" in self.preparation_states,
                )
                # ge-preparation drive pulse
                drive_pulse_ge = qt_ops.quantum_gate(qubit, f"X180_ge")
                excitation_ge_section.play(
                    signal=self.signal_name(f"drive", qubit),
                    pulse=drive_pulse_ge,
                )
                measure_play_after = f"{qubit.uid}_ge_excitation"
                excitation_sections += [excitation_ge_section]

                if prep_state == "f":
                    # create ef-preparation section
                    excitation_ef_section = Section(
                        uid=f"{qubit.uid}_ef_excitation",
                        alignment=SectionAlignment.RIGHT,
                        on_system_grid=True,
                        play_after=f"{qubit.uid}_ge_excitation",
                    )
                    # ef-preparation drive pulse
                    drive_pulse_ef = qt_ops.quantum_gate(qubit, f"X180_ef")
                    excitation_ef_section.play(
                        signal=self.signal_name(f"drive_ef", qubit),
                        pulse=drive_pulse_ef,
                    )
                    measure_play_after = f"{qubit.uid}_ef_excitation"
                    excitation_sections += [excitation_ef_section]

                freq_sweep_sections = freq_sweep.children
                # add play after excitation to measure_acquire section
                freq_sweep_sections[0].play_after = measure_play_after
                freq_sweep.children = excitation_sections + freq_sweep_sections
                acq_rt_loop.children[freq_sweep_idx] = freq_sweep

    def configure_experiment(self):
        for exp in self.experiments.values():
            exp.configure_experiment()

    def compile_experiment(self):
        for exp in self.experiments.values():
            exp.compile_experiment()

    def run_experiment(self):
        for exp in self.experiments.values():
            exp.run_experiment()
            # save experiments to the same directory, under the same timestamp
            exp.timestamp = self.timestamp
            exp.save_directory = self.save_directory

    def save_results(self):
        for state, exp in self.experiments.items():
            exp.save_results(filename_suffix=state)

    def analyse_experiment(self):
        ExperimentTemplate.analyse_experiment(self)
        for qubit in self.qubits:
            # all experiments have the same frequency axis
            exp = self.experiments["g"]
            handle = f"{exp.experiment_name}_{qubit.uid}"
            freqs = (
                exp.results.get_axis(handle)[0] + qubit.parameters.readout_lo_frequency
            )
            all_state_combinations = combinations(list(self.experiments), 2)
            s21_abs_distances = {"".join(sc): "" for sc in all_state_combinations}
            for i, states in enumerate(s21_abs_distances):
                s0, s1 = states
                s21_dist = abs(
                    self.experiments[s1].results.get_data(
                        f"{self.experiments[s1].experiment_name}_{qubit.uid}"
                    )
                    - self.experiments[s0].results.get_data(
                        f"{self.experiments[s0].experiment_name}_{qubit.uid}"
                    )
                )
                s21_abs_distances[states] = (s21_dist, np.argmax(s21_dist))

            s21_dist_sum = np.sum(
                [s21_dict[0] for s21_dict in s21_abs_distances.values()], axis=0
            )
            s21_abs_distances["sum"] = (s21_dist_sum, np.argmax(s21_dist_sum))
            self.analysis_results[qubit.uid]["s21_abs_distances"] = s21_abs_distances

            # plot S21 for each prep state
            fig_s21, ax_21 = plt.subplots()
            ax_21.set_xlabel("Readout Frequency, $f_{\\mathrm{RO}}$ (GHz)")
            ax_21.set_ylabel("Signal Magnitude, $|S_{21}|$ (a.u.)")
            ax_21.set_title(f"{self.timestamp}_{self.experiment_name}_{qubit.uid}")
            for state, exp in self.experiments.items():
                handle = f"{exp.experiment_name}_{qubit.uid}"
                freqs = (
                    exp.results.get_axis(handle)[0]
                    + qubit.parameters.readout_lo_frequency
                )
                data_mag = abs(exp.results.get_data(handle))
                ax_21.plot(freqs / 1e9, data_mag, label=state)
            ax_21.legend(frameon=False)

            # plot the S21 distances
            fig_s21_dist, ax_s21_dist = plt.subplots()
            ax_s21_dist.set_xlabel("Readout Frequency, $f_{\\mathrm{RO}}$ (GHz)")
            ax_s21_dist.set_ylabel(
                "Magnitude Signal Difference, $|\\Delta S_{21}|$ (a.u.)"
            )
            ax_s21_dist.set_title(
                f"{self.timestamp}_{self.experiment_name}_{qubit.uid}"
            )
            for states, (s21_dist, idx_max) in s21_abs_distances.items():
                max_s21_dist, max_freq = s21_dist[idx_max], freqs[idx_max]
                self.analysis_results[qubit.uid]["new_parameter_values"][
                    states
                ] = max_freq
                if states == "sum" and "f" not in self.preparation_states:
                    continue
                legend_label = (
                    f"{states}: $f_{{\\mathrm{{max}}}}$ = {max_freq / 1e9:.4f} GHz"
                )
                (line,) = ax_s21_dist.plot(freqs / 1e9, s21_dist, label=legend_label)
                # add point at max
                ax_s21_dist.plot(max_freq / 1e9, max_s21_dist, "o", c=line.get_c())
                # add vertical line at max
                ax_s21_dist.vlines(
                    max_freq / 1e9, min(s21_dist), max_s21_dist, colors=line.get_c()
                )
            ax_s21_dist.legend(frameon=False)

            # save figures and results
            if self.save:
                # Save the figures
                self.save_figure(fig_s21, qubit)
                fig_name = (
                    f"{self.timestamp}_{self.experiment_name}_S21_distances_{qubit.uid}"
                )
                self.save_figure(fig_s21_dist, qubit, figure_name=fig_name)
            if self.analysis_metainfo.get("show_figures", False):
                plt.show()
            plt.close(fig_s21)
            plt.close(fig_s21_dist)

    def update_qubit_parameters(self):
        for qubit in self.qubits:
            new_qb_pars = self.analysis_results[qubit.uid]["new_parameter_values"]
            if len(new_qb_pars) == 0:
                return
            qubit.parameters.readout_resonator_frequency = new_qb_pars["sum"]


class StateDiscrimination(ExperimentTemplate):
    fallback_experiment_name = "StateDiscrimination"
    valid_user_parameters = merge_valid_user_parameters(
        [
            dict(
                analysis_metainfo=[
                    "show_figures",
                ],
            ),
            ExperimentTemplate.valid_user_parameters,
        ]
    )

    def __init__(self, *args, signals=None, preparation_states=("g", "e"), **kwargs):
        self.preparation_states = preparation_states
        # Pass the preparation_states as the cal_states to the base class
        exp_metainfo = kwargs.get("experiment_metainfo", {})
        exp_metainfo.pop("cal_states", None)
        exp_metainfo["cal_states"] = self.preparation_states
        kwargs["experiment_metainfo"] = exp_metainfo
        # Add suffix to experiment name
        experiment_name = kwargs.get("experiment_name", self.fallback_experiment_name)
        experiment_name += f"_{''.join(self.preparation_states)}"
        kwargs["experiment_name"] = experiment_name

        # Set AveragingMode to SINGLE_SHOT
        acquisition_metainfo_user = kwargs.pop("acquisition_metainfo", dict())
        acquisition_metainfo = dict(averaging_mode=AveragingMode.SINGLE_SHOT)
        acquisition_metainfo.update(acquisition_metainfo_user)
        kwargs["acquisition_metainfo"] = acquisition_metainfo

        if signals is None:
            signals = ["drive", "measure", "acquire"]
        if "f" in self.preparation_states and "drive_ef" not in signals:
            signals += ["drive_ef"]

        super().__init__(
            *args, signals=signals, check_valid_user_parameters=False, **kwargs
        )

    def define_experiment(self):
        self.experiment.sections = []
        self.create_acquire_rt_loop()
        self.experiment.add(self.acquire_loop)
        for qubit in self.qubits:
            self.add_cal_states_sections(qubit)

    def analyse_experiment(self):
        super().analyse_experiment()
        for qubit in self.qubits:
            fig, ax = plt.subplots()
            shots = {}
            for i, ps in enumerate(self.preparation_states):
                handle = f"{self.experiment_name}_{qubit.uid}_cal_trace_{ps}"
                shots[ps] = self.results.get_data(handle)
                ax.scatter(np.real(shots[ps]), np.imag(shots[ps]), c=f"C{i}", label=ps)
                # plot mean point
                mean_state = np.mean(shots[ps])
                ax.plot(
                    np.real(mean_state), np.imag(mean_state), "o", mfc=f"C{i}", mec="k"
                )

            # compute the distances between the mean of the points for each state
            all_state_combinations = combinations(self.preparation_states, 2)
            distances_means = {"".join(sc): "" for sc in all_state_combinations}
            textstr = ""
            for i, states in enumerate(distances_means):
                s0, s1 = states
                distances_means[states] = abs(np.mean(shots[s0]) - np.mean(shots[s1]))
                textstr += f"dist({s0},{s1}): {distances_means[states]:.2f}\n"
            distances_means["sum"] = np.sum(list(distances_means.values()))
            self.analysis_results[qubit.uid]["distances_means"] = distances_means
            textstr += f"sum: {distances_means['sum']:.2f}"
            ax.text(1.025, 0.5, textstr, ha="left", va="center", transform=ax.transAxes)
            ax.set_xlabel("Real Signal Component, $V_I$ (a.u.)")
            ax.set_ylabel("Imaginary Signal Component, $V_Q$ (a.u.)")
            ax.set_title(f"{self.timestamp}_{self.experiment_name}_{qubit.uid}")
            ax.legend(frameon=False)
            if self.save:
                # Save the figures
                self.save_figure(fig, qubit)
            if self.analysis_metainfo.get("show_figures", False):
                plt.show()
            plt.close(fig)
