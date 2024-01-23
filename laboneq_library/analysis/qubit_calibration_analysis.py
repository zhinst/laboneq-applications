import numpy as np
import uncertainties as unc
import matplotlib.pyplot as plt

import logging

log = logging.getLogger(__name__)

from laboneq_library.analysis import analysis_helpers as ana_hlp
from laboneq.analysis import fitting as fit_mods
from laboneq_library.analysis.analysis_library import (
    AnalysisTemplate,
    RawDataProcessingMixin,
    SkipStepException,
    AnalysisStep,
    StandAloneStep,
    LmfitStep,
)

# class ReparkingRamsey(AnalysisTemplate):
#     analysis_steps = [
#         RamseyAnalysis,
#         self.run_fitting,
#         self.extract_new_parameter_values,
#         self.extract_old_parameter_values]


class SingleQubitTuneupAnalysis(AnalysisTemplate, RawDataProcessingMixin):
    fallback_analysis_name = "SingleQubitTuneupAnalysis"
    swpts_scaling_factor = 1
    xaxis_label = None

    def __init__(self, *args, **kwargs):
        analysis_steps = [
            self.extract_and_process_data_cal_states,
            self.run_fitting,
            self.extract_old_parameter_values,
            self.extract_new_parameter_values,  # old value might be used here
            self.run_plotting,
        ]
        analysis_steps_dependency = np.arange(len(analysis_steps))
        ana_metainfo = kwargs.get("analysis_metainfo", {})
        self.cal_states = ana_metainfo.get("cal_states", "ge")
        self.tr_to_calib = ana_metainfo.get("transition_to_calibrate", "ge")

        super().__init__(
            *args,
            analysis_steps=analysis_steps,
            analysis_steps_dependency=analysis_steps_dependency,
            **kwargs,
        )

    def run_fitting(self, qubit):
        if not self.do_fitting:
            raise SkipStepException

    def extract_new_parameter_values(self, qubit):
        if not self.do_fitting:
            raise SkipStepException

    def extract_old_parameter_values(self, qubit):
        pass

    def run_plotting(self, qubit):
        if not self.do_plotting:
            raise SkipStepException

        handle = f"{self.experiment_name}_{qubit.uid}"
        data_dict = self.analysis_results["rotated_data"][qubit.uid]
        num_cal_traces = data_dict["num_cal_traces"]
        do_pca = self.analysis_metainfo.get("do_pca", False)

        fig, ax = plt.subplots()
        xlabel = self.xaxis_label
        if xlabel is None:
            xlabel = self.results.get_axis_name(handle)[0]
        ax.set_xlabel(xlabel)
        ax.set_ylabel(
            "Principal Component (a.u)"
            if (num_cal_traces == 0 or do_pca)
            else f"$|{self.cal_states[-1]}\\rangle$-State Population"
        )
        ax.set_title(f"{self.timestamp}_{handle}")

        self.plot_fitted_data(qubit, ax)
        if self.do_fitting:
            self.plot_fit(qubit, ax)
            self.plot_textbox(qubit, ax)

        self.figures_to_save += [(fig, qubit, None)]

    def plot_fitted_data(self, qubit, ax):
        data_dict = self.analysis_results["rotated_data"][qubit.uid]
        ax.plot(
            data_dict["sweep_points_w_cal_traces"] * self.swpts_scaling_factor,
            data_dict["data_rotated_w_cal_traces"],
            "o",
            zorder=2,
        )

    def plot_fit(self, qubit, ax):
        pass

    def plot_textbox(self, qubit, ax):
        pass

    def create_text_string(self, qubit, parameter_info):
        """
        Create and add a textbox to an axis.

        The textbox is created from the new and old qubit-parameters values.

        Args:
            qubit: instance of a qubit class
            ax: instance of an axis
            parameter_info: list of tuples where
                - the first entry in the tuple is either
                    - a string with the name under which the qubit parameter has been
                    stored in new_parameter_values/old_parameter_values
                    - a float giving the value of the parameter
                    - a tuple giving the value and stderr of the parameter
                - the second entry in the tuple is the parameter label to be shown in
                    the textbox (usually latex)
                - the third entry in the tuple gives information about the unit.
                    The entry has the following form:
                    - string: the unit for both the parameter value and its stderr; ex: 'GHz'
                    - tuple: the units for the parameter value and its stderr; ex: ('GHz', 'MHz')
                - the fourth entry in the tuple gives information about the scaling
                    factors of the parameters. The entry has the following form:
                    - float: the scaling factor for both the parameter value and its
                        stderr; ex: 1e-6
                    - tuple: the scaling factors for the parameter value and its stderr;
                        ex: (1e-9, 1e-6)
                - the fifth entry in the tuple gives information about the precision
                    factors of the parameters, i.e. the number of decimal points to show
                    in the textbox. The entry has the following form:
                    - integer: the precision for both the parameter value and its
                        stderr; ex: 4
                    - tuple: the precision for the parameter value and its stderr;
                        ex: (6, 4)

                Ex: [('resonance_frequency', New qubit frequency, ('GHz', MHz))]
                    [('amplitude_pi', $A_{\\pi}$, '',)]
                    [(1.1e6, Introduced detuning, 'MHz')]

        Returns:
            the text string
        """

        # units_to_scaling_factors = {"GHz": 1e-9, "MHz": 1e-3, "kHz": 1e-3,
        #                             "ns": 1e9, "$\\mu$s": 1e-6, "us": 1e-6,
        #                             "mV": 1e3, "uV": 1e6, "$\\mu$V": 1e6}
        textstr = ""
        for i, param_info in enumerate(parameter_info):
            (
                param_main,
                param_label,
                unit,
                scaling_factors,
                precision_factors,
            ) = param_info

            if isinstance(unit, str):
                value_unit, err_unit = unit, unit
            else:
                value_unit, err_unit = unit[0], unit[1]

            if hasattr(scaling_factors, "__iter__"):
                sf_val, sf_err = scaling_factors
            else:
                sf_val, sf_err = scaling_factors, scaling_factors

            if hasattr(precision_factors, "__iter__"):
                pf_val, pf_err = precision_factors
            else:
                pf_val, pf_err = precision_factors, precision_factors

            if not hasattr(param_main, "__iter__"):
                # param_main is the value of the parameter
                new_param_val, old_param_val = param_main, None
            elif isinstance(param_main, str):
                # param_main is the name of the parameter as stored in
                # new_parameter_values/old_parameter_values
                new_param_val = self.analysis_results["new_parameter_values"][
                    qubit.uid
                ][f"{param_main}"]
                old_param_val = self.analysis_results["old_parameter_values"][
                    qubit.uid
                ].get(f"{param_main}", None)
            else:
                # param_main is a tuple with (val, stderr)
                new_param_val, old_param_val = param_main, None

            if i > 0:
                textstr += "\n"
            if hasattr(new_param_val, "__iter__"):
                # new_param_val is a tuple with (value, stderr)
                textstr += (
                    f"{param_label}: {new_param_val[0]*sf_val:.{pf_val}f} {value_unit} "
                    f"$\\pm$ {new_param_val[1]*sf_err:.{pf_err}f} {err_unit}"
                )
            else:
                # it is just the value
                textstr += (
                    f"{param_label}: {new_param_val*sf_val:.{pf_val}f} {value_unit}"
                )
            if old_param_val is not None:
                textstr += f"\nOld {param_label}: {old_param_val*sf_val:.{pf_val}f} {value_unit}"

        return textstr


class AmplitudeRabiAnalysis(SingleQubitTuneupAnalysis):
    fallback_analysis_name = "AmplitudeRabiAnalysis"
    swpts_scaling_factor = 1
    xaxis_label = "Amplitude Scaling Factor"

    def run_fitting(self, qubit):
        super().run_fitting(qubit)
        data_dict = self.analysis_results["rotated_data"][qubit.uid]
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
        self.analysis_results["fit_results"][qubit.uid] = fit_res

    def extract_new_parameter_values(self, qubit):
        super().extract_new_parameter_values(qubit)
        data_dict = self.analysis_results["rotated_data"][qubit.uid]
        swpts_to_fit = data_dict["sweep_points"]
        fit_res = self.analysis_results["fit_results"][qubit.uid]

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
        self.analysis_results["new_parameter_values"][qubit.uid] = {
            f"{self.tr_to_calib}_amplitude_pi": np.array(
                [
                    pi_amp.nominal_value,
                    pi_amp.std_dev,
                ]
            ),
            f"{self.tr_to_calib}_amplitude_pi2": np.array(
                [
                    pi2_amp.nominal_value,
                    pi2_amp.std_dev,
                ]
            ),
            f"{self.tr_to_calib}_pi_amps": [pia.nominal_value for pia in pi_amps],
            f"{self.tr_to_calib}_pi2_amps": [pi2a.nominal_value for pi2a in pi_amps],
        }

    def extract_old_parameter_values(self, qubit):
        super().extract_old_parameter_values(qubit)
        old_pi_amp = (
            qubit.parameters.drive_parameters_ef["amplitude_pi"]
            if "f" in self.tr_to_calib
            else qubit.parameters.drive_parameters_ge["amplitude_pi"]
        )
        old_pi2_amp = (
            qubit.parameters.drive_parameters_ef["amplitude_pi2"]
            if "f" in self.tr_to_calib
            else qubit.parameters.drive_parameters_ge["amplitude_pi2"]
        )
        self.analysis_results["old_parameter_values"][qubit.uid] = {
            f"{self.tr_to_calib}_amplitude_pi": old_pi_amp,
            f"{self.tr_to_calib}_amplitude_pi2": old_pi2_amp,
        }

    def plot_fit(self, qubit, ax):
        fit_res = self.analysis_results["fit_results"][qubit.uid]
        data_dict = self.analysis_results["rotated_data"][qubit.uid]
        swpts_to_fit = data_dict["sweep_points"]
        swpts_fine = np.linspace(swpts_to_fit[0], swpts_to_fit[-1], 501)
        ax.plot(
            swpts_fine,
            fit_res.model.func(swpts_fine, **fit_res.best_values),
            "r-",
            zorder=1,
        )
        pi_amp = self.analysis_results["new_parameter_values"][qubit.uid][
            f"{self.tr_to_calib}_amplitude_pi"
        ]
        plt.plot(
            pi_amp[0],
            fit_res.model.func(pi_amp[0], **fit_res.best_values),
            "sk",
            zorder=3,
            markersize=plt.rcParams["lines.markersize"] + 1,
        )
        pi2_amp = self.analysis_results["new_parameter_values"][qubit.uid][
            f"{self.tr_to_calib}_amplitude_pi2"
        ]
        plt.plot(
            pi2_amp[0],
            fit_res.model.func(pi2_amp[0], **fit_res.best_values),
            "sk",
            zorder=3,
            markersize=plt.rcParams["lines.markersize"] + 1,
        )

    def plot_textbox(self, qubit, ax):
        # Add text box
        parameter_info = [
            (f"{self.tr_to_calib}_amplitude_pi2", "$A_{\\pi/2}$", "", (1, 1), (4, 4)),
        ]
        textstr = self.create_text_string(qubit, parameter_info)
        ax.text(0, -0.15, textstr, ha="left", va="top", transform=ax.transAxes)
        parameter_info = [
            (f"{self.tr_to_calib}_amplitude_pi", "$A_{\\pi}$", "", (1, 1), (4, 4)),
        ]
        textstr = self.create_text_string(qubit, parameter_info)
        ax.text(0.69, -0.15, textstr, ha="left", va="top", transform=ax.transAxes)


class RamseyAnalysis(SingleQubitTuneupAnalysis):
    fallback_analysis_name = "RamseyAnalysis"
    swpts_scaling_factor = 1e6
    xaxis_label = "Pulse Separation, $\\tau$ ($\\mu$s)"

    def run_fitting(self, qubit):
        super().run_fitting(qubit)
        data_dict = self.analysis_results["rotated_data"][qubit.uid]
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
            ana_hlp.oscillatory_decay_flexible,
            swpts_to_fit,
            data_to_fit,
            param_hints=param_hints,
        )
        self.analysis_results["fit_results"][qubit.uid] = fit_res

    def extract_old_parameter_values(self, qubit):
        super().extract_old_parameter_values(qubit)
        old_qb_freq = (
            qubit.parameters.resonance_frequency_ef
            if "f" in self.tr_to_calib
            else qubit.parameters.resonance_frequency_ge
        )
        self.analysis_results["old_parameter_values"][qubit.uid] = {
            "resonance_frequency": old_qb_freq
        }

    def extract_new_parameter_values(self, qubit):
        super().extract_new_parameter_values(qubit)
        fit_res = self.analysis_results["fit_results"][qubit.uid]
        t2_star = fit_res.best_values["decay_time"]
        t2_star_err = fit_res.params["decay_time"].stderr
        freq_fit = fit_res.best_values["frequency"]
        freq_fit_err = fit_res.params["frequency"].stderr

        old_qb_freq = self.analysis_results["old_parameter_values"][qubit.uid][
            "resonance_frequency"
        ]
        introduced_detuning = self.experiment_metainfo["detuning"][qubit.uid]
        new_qb_freq = old_qb_freq + introduced_detuning - freq_fit
        self.analysis_results["new_parameter_values"][qubit.uid] = {
            "resonance_frequency": np.array([new_qb_freq, freq_fit_err]),
            "T2_star": np.array([t2_star, t2_star_err]),
        }

    def plot_fit(self, qubit, ax):
        fit_res = self.analysis_results["fit_results"][qubit.uid]
        data_dict = self.analysis_results["rotated_data"][qubit.uid]
        swpts_to_fit = data_dict["sweep_points"]
        swpts_fine = np.linspace(swpts_to_fit[0], swpts_to_fit[-1], 501)
        ax.plot(
            swpts_fine * 1e6,
            fit_res.model.func(swpts_fine, **fit_res.best_values),
            "r-",
            zorder=1,
        )

    def plot_textbox(self, qubit, ax):
        new_qb_freq = self.analysis_results["new_parameter_values"][qubit.uid][
            "resonance_frequency"
        ]
        old_qb_freq = self.analysis_results["old_parameter_values"][qubit.uid][
            "resonance_frequency"
        ]
        introduced_detuning = self.experiment_metainfo["detuning"][qubit.uid]
        fit_res = self.analysis_results["fit_results"][qubit.uid]
        freq_fit = fit_res.best_values["frequency"]
        freq_fit_err = fit_res.params["frequency"].stderr
        parameter_info = [
            (
                "resonance_frequency",
                "New qubit frequency",
                ("GHz", "MHz"),
                (1e-9, 1e-6),
                (6, 4),
            ),
            (
                (new_qb_freq[0] - old_qb_freq, new_qb_freq[1]),
                "Diff new-old qubit frequency",
                ("MHz", "kHz"),
                (1e-6, 1e-3),
                (6, 4),
            ),
            (introduced_detuning, "Introduced detuning", "MHz", 1e-6, 2),
            (
                (freq_fit, freq_fit_err),
                "Fitted frequency",
                ("MHz", "kHz"),
                (1e-6, 1e-3),
                (6, 4),
            ),
            ("T2_star", "$T_2^*$", ("", "$\\mu$s"), 1e6, 4),
        ]
        textstr = self.create_text_string(qubit, parameter_info)
        ax.text(0, -0.15, textstr, ha="left", va="top", transform=ax.transAxes)
