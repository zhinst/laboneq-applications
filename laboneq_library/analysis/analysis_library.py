import json
import os
import time
import lmfit
import dill as pickle
import matplotlib.pyplot as plt
from functools import update_wrapper, partial

import logging
import traceback

log = logging.getLogger(__name__)

from laboneq_library import loading_helpers as load_hlp
from laboneq_library.analysis import analysis_helpers as ana_hlp


class SkipStepException(Exception):
    """Raised when an analysis step is to be skipped."""

    pass


class AnalysisStep:
    def __init__(self, func):
        update_wrapper(self, func)  # added to make it work with decorating methods
        self.function = func
        self.input_arguments = None
        self.result = None

    def __get__(self, obj, objtype):
        """Support instance methods."""
        return partial(self.__call__, obj)

    # def __call__(self, **kwargs):
    #     self.input_arguments = kwargs
    #     self.result = self.function(**kwargs)
    #     return self
    def __call__(self, obj, **kwargs):
        self.input_arguments = kwargs
        self.result = self.function(obj, **kwargs)
        return self

    @property
    def input_parameters(self):
        return self.input_arguments

    @property
    def output(self):
        return self.result


class LmfitStep(AnalysisStep):
    def __call__(self, obj, **kwargs):
        super().__call__(obj, **kwargs)
        self.result = ana_hlp.fit_data_lmfit(
            self.input_parameters["model"],
            self.input_parameters["independent_variable"],
            self.input_parameters["data_to_fit"],
            param_hints=self.input_parameters["param_hints"],
        )
        return self


class StandAloneStep(AnalysisStep):
    def __call__(self, obj, **kwargs):
        self.input_arguments = {}
        self.result = self.function(obj)
        return self


class AnalysisTemplate:
    fallback_analysis_name = "Analysis"
    analysis_results = dict(
        new_parameter_values=dict(),
        old_parameter_values=dict(),
        fit_results=dict(),
    )
    figures_to_save = []  # {qubit.uid: [(fig_name, fig)]}

    def __init__(
        self,
        qubits,
        results,
        experiment_metainfo=None,
        analysis_name=None,
        analysis_metainfo=None,
        analysis_steps=None,
        analysis_steps_dependency=None,
        save_directory=None,
        data_directory=None,
        save=True,
        run=False,
    ):
        self.qubits = qubits
        if self.qubits is None:
            raise ValueError("Please provide a list of qubits.")
        self.results = results
        if self.results is None:
            raise ValueError("Please provide an acquired_results object.")
        self.experiment_metainfo = experiment_metainfo
        if self.experiment_metainfo is None:
            self.experiment_metainfo = {}
        self.experiment_name = self.experiment_metainfo.get("experiment_name", None)
        if self.experiment_name is None:
            raise ValueError("Please provide experiment_name in experiment_metainfo.")

        self.analysis_name = analysis_name
        if self.analysis_name is None:
            self.analysis_name = self.fallback_analysis_name
        self._analysis_steps = analysis_steps
        self._analysis_steps_dependency = analysis_steps_dependency
        if self._analysis_steps is None and self._analysis_steps_dependency is None:
            raise ValueError(
                "Please provide analysis_steps_dependency, or do not "
                "specify 'analysis_steps'."
            )

        self.analysis_metainfo = analysis_metainfo
        if self.analysis_metainfo is None:
            self.analysis_metainfo = {}
        self.do_fitting = self.analysis_metainfo.get("do_fitting", True)
        self.do_plotting = self.analysis_metainfo.get("do_plotting", True)
        # processing_steps: project_cal_states, do_pca, (future: filter, average
        self.save_directory = save_directory
        self.timestamp = load_hlp.get_timestamp_from_experiment_directory(
            self.save_directory
        )
        self.data_directory = data_directory
        self.save = save

        self.run = run
        if self.run:
            self.autorun()

    def create_analysis_label(self):
        if len(self.qubits) <= 5:
            qb_names_suffix = f'{"".join([qb.uid for qb in self.qubits])}'
        else:
            qb_names_suffix = f"{len(self.qubits)}qubits"
        self.analysis_label = f"{self.analysis_name}_{qb_names_suffix}"

    def generate_timestamp_save_directory(self):
        # create analysis timestamp
        self.timestamp = str(time.strftime("%Y%m%d_%H%M%S"))
        if self.data_directory is not None:
            # create experiment save_directory
            self.save_directory = os.path.abspath(
                os.path.join(
                    self.data_directory,
                    f"{self.timestamp[:8]}",
                    f"{self.timestamp[-6:]}_{self.analysis_label}",
                )
            )

    def create_save_directory(self):
        # create the save_directory inside self.data_directory
        if self.save_directory is not None and not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)

    def autorun(self):
        try:
            for qubit in self.qubits:
                for idx in self._analysis_steps_dependency:
                    try:
                        if hasattr(idx, "__iter__"):
                            ana_step_input = self._analysis_steps[idx[0]]
                            ana_step_to_run = self._analysis_steps[idx[1]]
                            if ana_step_input.output is None:
                                raise ValueError(
                                    f"Analysis step {ana_step_input.name} "
                                    f"(idx={idx[0]}) cannot be used as an input to "
                                    f"analysis step {ana_step_to_run.name} (idx={idx[1]}) "
                                    f"because the former was not run."
                                )
                                ana_step_to_run(ana_step_input.output)
                        else:
                            # analysis step without input parameters
                            self._analysis_steps[idx](qubit)
                    except SkipStepException:
                        continue
        except Exception:
            log.error("Unhandled error during AnalysisTemplate!")
            log.error(traceback.format_exc())
        # self.process_data()
        # if self.do_fitting:
        #     self.run_fitting()
        #     self.extract_new_parameter_values()
        # if self.do_plotting:
        #     self.run_plotting()
        if self.save:
            self.save_analysis_results()
            for save_fig_args in self.figures_to_save:
                self.save_figure(*save_fig_args)
            if self.analysis_metainfo.get("show_figures", False):
                plt.show()
            for save_fig_args in self.figures_to_save:
                plt.close(save_fig_args[0])

    def save_figure(self, fig, qubit, figure_name=None):
        self.create_save_directory()
        fig_name = self.analysis_metainfo.get("figure_name", figure_name)
        if fig_name is None:
            fig_name = f"{self.timestamp}_{self.analysis_name}_{qubit.uid}"
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
        new_qb_params_exist = len(self.analysis_results["new_parameter_values"]) > 0
        old_qb_params_exist = len(self.analysis_results["old_parameter_values"]) > 0
        fit_results_exist = len(self.analysis_results["fit_results"]) > 0
        other_ana_res_exist = len(self.analysis_results) > 3

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
                fit_results = self.analysis_results["fit_results"]
                # Save fit results into a json file
                fit_res_file = os.path.abspath(
                    os.path.join(
                        self.save_directory,
                        f"{self.timestamp}_fit_results{filename_suffix}.json",
                    )
                )
                if isinstance(fit_results, lmfit.model.ModelResult):
                    with open(fit_res_file, "w") as file:
                        json.dump(
                            ana_hlp.flatten_lmfit_modelresult(fit_results),
                            file,
                            indent=2,
                        )
                elif isinstance(fit_results, dict):
                    fit_results_to_save_json = dict()
                    for qbuid, fit_res in fit_results.items():
                        # Convert lmfit results into a dictionary that can be saved
                        # as json
                        if isinstance(fit_res, dict):
                            fit_results_to_save_json[qbuid] = {}
                            for k, fr in fit_res.items():
                                fit_results_to_save_json[qbuid][
                                    k
                                ] = ana_hlp.flatten_lmfit_modelresult(fr)
                        else:
                            fit_results_to_save_json[
                                qbuid
                            ] = ana_hlp.flatten_lmfit_modelresult(fit_res)

                    with open(fit_res_file, "w") as file:
                        json.dump(fit_results_to_save_json, file, indent=2)

            # Save analysis_results pickle file
            ana_res_file = os.path.abspath(
                os.path.join(
                    self.save_directory,
                    f"{self.timestamp}_analysis_results{filename_suffix}.p",
                )
            )
            with open(ana_res_file, "wb") as f:
                pickle.dump(self.analysis_results, f)


class RawDataProcessingMixin:
    def extract_and_process_data_cal_states(self, qubit):
        if "rotated_data" not in self.analysis_results:
            self.analysis_results["rotated_data"] = {}
        handle = f"{self.experiment_name}_{qubit.uid}"
        do_pca = self.analysis_metainfo.get("do_pca", False)
        data_dict = ana_hlp.extract_and_rotate_data_1d(
            self.results, handle, cal_states=self.cal_states, do_pca=do_pca
        )
        self.analysis_results["rotated_data"][qubit.uid] = data_dict


# class AnalysisStep:
#     def __init__(self, *args, **kwargs):
#         self.input_data = None
#         self.output_data = None
#
#     def __call__(self, *args, **kwargs):
#         self.execute()
#
#     def execute(self):
#         pass
#
#     # def pre(self):
#     #
#     # def analyze(self):
#     #     self.function_to_be_fitted(self.extracted_and_processed_data)
#     #
#     # def post(self):
#
#
# class DataProcessingStep(AnalysisStep):
#     def __init__(self, function):
#         super().__init__()
#         self.function = function
#
#
# class FittingStep(AnalysisStep):
#     def __init__(self, model, param_hints):
#         super().__init__()
#         self.model = model
#         self.param_hints = param_hints
#
#     def execute(self, *args, **kwargs):
#         self.output_data = ana_hlp.fit_data_lmfit(
#             self.model,
#             self.input_data["independent_variable"],
#             self.input_data["data_to_fit"],
#             param_hints=self.param_hints,
#         )
#
#
# class PlottingStep(AnalysisStep):
#     pass
