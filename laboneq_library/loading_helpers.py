import json
import numpy as np
import dill as pickle
import os

from laboneq.dsl.result.acquired_result import AcquiredResult, AcquiredResults
from laboneq.dsl.serialization import Serializer
from laboneq_library import calibration_helpers as calib_hlp
from laboneq.simple import *  # noqa: F403
from ruamel.yaml import YAML

ryaml = YAML()

import logging

log = logging.getLogger(__name__)


def get_latest_experiment_directory(data_directory):
    """
    Returns the last data folder in experiment_directory.

    Example: data_directory contains the following folders:
    ["20231128", "20231127", "20231126"]. This function returns the most recent
    folder inside data_directory/20231128, as sorted by the timestamp in the
    names of the folders.

    Args:
        data_directory: directory where the all the experiment data is saved

    Returns:
        the latest data folder
    """
    latest_folder = None
    day_folders = os.listdir(data_directory)
    day_folders.sort()
    for day_folder in day_folders[::-1]:
        ts_folders = os.listdir(data_directory + f"\\{day_folder}")
        if len(ts_folders) == 0:
            continue
        else:
            ts_folders.sort()
            ts_folder = ts_folders[-1]
            latest_folder = data_directory + f"\\{day_folder}\\{ts_folder}"
            break
    if latest_folder is None:
        raise ValueError(f"Did not find any measurement folders in {data_directory}.")
    return latest_folder


def get_timestamp_from_experiment_directory(experiment_directory):
    split_save_dir = experiment_directory.split("\\")
    return "_".join([split_save_dir[-2], split_save_dir[-1][:6]])


# helpers to get full file paths to common files inside an experiment directory
def get_measurement_setup_file_path(experiment_directory, file_extension="json"):
    """
    Get the full file path to a measurement_setup file in experiment_directory,
    with the extension file_extension.

    Args:
        experiment_directory: path to the directory where the measurement data is saved
        file_extension: extension of the measurement_setup file

    Returns:
        the full path to a measurement_setup file
    """
    msmt_setup_fp = [
        f
        for f in os.listdir(experiment_directory)
        if f"measurement_setup" in f and f.endswith(file_extension)
    ]
    if len(msmt_setup_fp) == 0:
        raise FileNotFoundError(
            f"The data folder {experiment_directory} does not contain a .{file_extension} "
            f"file with the name 'measurement_setup.'"
        )
    elif len(msmt_setup_fp) > 1:
        raise ValueError(
            f"More than one .{file_extension} file containing the name "
            f"'measurement_setup' was found in {experiment_directory}. Unclear which "
            f"one to take."
        )
    else:
        msmt_setup_fp = msmt_setup_fp[0]
    return experiment_directory + f"\\{msmt_setup_fp}"


def get_results_file_path(experiment_directory, file_extension="json"):
    """
    Get the full file path to a result file  in experiment_directory,
    with the extension file_extension.

    Args:
        experiment_directory: path to the directory where the measurement data is saved
        file_extension: extension of the results file

    Returns:
        the full path to a results file
    """
    results_fp = [
        f
        for f in os.listdir(experiment_directory)
        if f"results.{file_extension}" in f
        and "fit" not in f
        and "acquired" not in f
        and "analysis" not in f
    ]
    if len(results_fp) == 0:
        raise FileNotFoundError(
            f"The data folder {experiment_directory} does not contain a .{file_extension} "
            f"file with the name 'results.'"
        )
    elif len(results_fp) > 1:
        raise ValueError(
            f"More than one .{file_extension} file containing the name "
            f"'results' was found in {experiment_directory}. Unclear which "
            f"one to take."
        )
    else:
        results_fp = results_fp[0]
    return experiment_directory + f"\\{results_fp}"


def get_acquired_results_file_path(experiment_directory,
                                   filename_to_match="acquired_results",
                                   file_extension="p"):
    """
    Get the full file path to an acquired_results file in experiment_directory,
    with the extension file_extension.

    Args:
        experiment_directory: path to the directory where the measurement data is saved
        filename_to_match: string to match in the filename
        file_extension: extension of the acquired_results file

    Returns:
        the full path to an acquired_results file
    """
    aq_results_fp = [
        f
        for f in os.listdir(experiment_directory)
        if f"{filename_to_match}.{file_extension}" in f
    ]
    if len(aq_results_fp) == 0:
        raise FileNotFoundError(
            f"The data folder {experiment_directory} does not contain a .{file_extension} "
            f"file with the name 'acquired_results.'"
        )
    elif len(aq_results_fp) > 1:
        raise ValueError(
            f"More than one .{file_extension} file containing the name "
            f"'acquired_results' was found in {experiment_directory}. Unclear which "
            f"one to take."
        )
    else:
        aq_results_fp = aq_results_fp[0]
    return experiment_directory + f"\\{aq_results_fp}"


def get_analysis_results_file_path(experiment_directory, file_extension="p"):
    """
    Get the full file path to an analysis_results file in experiment_directory,
    with the extension file_extension.

    Args:
        experiment_directory: path to the directory where the measurement data is saved
        file_extension: extension of the analysis_results file

    Returns:
        the full path to an analysis_results file
    """
    ana_setup_fp = [
        f
        for f in os.listdir(experiment_directory)
        if f"analysis_results" in f and f.endswith(file_extension)
    ]
    if len(ana_setup_fp) == 0:
        raise FileNotFoundError(
            f"The data folder {experiment_directory} does not contain a .{file_extension} "
            f"file with the name 'analysis_results.'"
        )
    elif len(ana_setup_fp) > 1:
        raise ValueError(
            f"More than one .{file_extension} file containing the name "
            f"'measurement_setup' was found in {experiment_directory}. Unclear which "
            f"one to take."
        )
    else:
        ana_setup_fp = ana_setup_fp[0]
    return experiment_directory + f"\\{ana_setup_fp}"


# loading
def load_measurement_setup_from_experiment_directory(
    experiment_directory, before_experiment=True
):
    """
    Load a DeviceSetup from the experiment_directory.

    Searched for a filename that contains "measurement_setup.json" inside
    experiment_directory.

    Args:
        experiment_directory: path to the directory where the measurement data is saved
        before_experiment: whether to load the setup from the file saves before
            (True) or after (False) running the experiment, which might update
            the setup. If True, the setup is loaded from the file inside
            experiment_directory which contains the name "measurement_setup". If False,
            the setup is loaded from the results saved in the file inside
            experiment_directory which contains the name "results"

    Returns:
        instance of DeviceSetup
    """
    if before_experiment:
        try:
            msmt_setup_fp = get_measurement_setup_file_path(
                experiment_directory, file_extension="json"
            )
            measurement_setup = DeviceSetup.load(msmt_setup_fp)
        except Exception:
            log.warning(
                "Could not deserialise the measurement_setup object. "
                "Loading the measurement_setup from pickle."
            )
            msmt_setup_fp = get_measurement_setup_file_path(
                experiment_directory, file_extension="p"
            )
            measurement_setup = pickle.load(open(msmt_setup_fp, "rb"))
    else:
        results = load_results_from_experiment_directory(experiment_directory)
        measurement_setup = results.device_setup
    return measurement_setup


def load_acquired_results_from_experiment_directory(
        experiment_directory, filename_to_match=None, file_extension="json"):
    """
    Load an AcquiredResults object from a file in the experiment_directory, which has
    the extension file_extension.

    Searches for a filename that contains f"analysis_results.{file_extension}" inside
    experiment_directory.

    Args:
        experiment_directory: path to the directory where the measurement data is saved
        filename_to_match: string to match in the filename
        file_extension: extension of the acquired_results file


    Returns:
        instance of AcquiredResults
    """

    acquired_results_fp = get_acquired_results_file_path(
        experiment_directory, filename_to_match, file_extension
    )
    if file_extension == "p":
        acquired_results = pickle.load(open(acquired_results_fp, "rb"))
    elif file_extension == "json":
        try:
            analysis_results_fp = get_analysis_results_file_path(
                experiment_directory, file_extension="json")
            acquired_results = Serializer.from_json_file(analysis_results_fp, type_hint=AcquiredResults)
        except Exception:
            log.warning(
                "Could not deserialise the acquired_results object or no "
                "acquired_results was found. Loading the acquired results "
                "from the results json file."
            )
            results_fp = get_results_file_path(experiment_directory, file_extension="json")
            acquired_results = load_acquired_results_from_results_json(results_fp)
    else:
        raise ValueError(f"Unrecognised file extension {file_extension}. Currently only "
                         f"'json' or 'p' (pickle) are supported.")
    return acquired_results


def load_acquired_results_from_results_json(
    json_file_path=None, experiment_directory=None
):
    """
    Create an instance of AcquiredResults from the acquired results loaded from a
    results.json file saved by LabOne Q.

    Args:
        json_file_path: full file path to the results.json file
        experiment_directory: path to the directory where the measurement data is saved
            and which contains a results.json file

    Returns:
        An instance of AcquiredResults
    """

    # extract results from json file
    if json_file_path is not None:
        results = json.load(open(json_file_path))
    else:
        if experiment_directory is None:
            raise ValueError(
                "Please provide either the json_file_path or the experiment_directory."
            )

        results_fp = get_results_file_path(experiment_directory, file_extension="json")
        results = json.load(open(results_fp))

    # decode acquired results
    acquired_results_loaded = results["results"]["acquired_results"]
    acquired_results = {}
    for handle in acquired_results_loaded:
        acquired_results[handle] = AcquiredResult()
        for key, value in acquired_results_loaded[handle].items():
            if key == "__type":
                continue
            elif key == "axis":
                axis_list = []
                for d in value:
                    if isinstance(d, list):
                        for d_sub in d:
                            axis_list += [[np.array(d_sub["real_data"])]]
                    else:
                        axis_list += [np.array(d["real_data"])]
                setattr(acquired_results[handle], key, axis_list)
            elif key == "data":
                try:
                    data_strings = value["complex_data"]
                    setattr(
                        acquired_results[handle],
                        key,
                        np.array(list(map(complex, data_strings))),
                    )
                except KeyError:
                    setattr(
                        acquired_results[handle],
                        key,
                        value["real"] + 1j * value["imag"],
                    )
            else:
                setattr(acquired_results[handle], key, value)
    return AcquiredResults(acquired_results)


def load_results_from_experiment_directory(experiment_directory):
    """
    Load a Results or AnalysisResults object from the experiment_directory.

    Args:
        experiment_directory: path to the directory where the measurement data is saved

    Returns:
        instance of Results or instance of AcquiredResults in case the Results could not
        be deserialised
    """

    results_fp = get_results_file_path(experiment_directory, file_extension="json")
    try:
        results = Results.load(results_fp)
    except Exception:
        log.warning(
            "Could not deserialise the result object. Loading the acquired_results."
        )
        results = load_acquired_results_from_experiment_directory(
            experiment_directory, file_extension="json")
    return results


def load_qubit_parameters_from_yaml(filename="./qubit_parameters.yaml"):
    with open(filename) as f:
        calib_file = f.read()
    qubit_parameters = ryaml.load(calib_file)
    return qubit_parameters


def load_qubit_parameters_from_json(experiment_directory=None, full_filepath=None):
    if full_filepath is None:
        if experiment_directory is None:
            raise ValueError(
                "Please provide either experiment_directory or full_filepath."
            )
        full_filepath = [
            fn
            for fn in os.listdir(experiment_directory)
            if "qubit_parameters.json" in fn
        ]
        if len(full_filepath) == 0:
            raise FileNotFoundError(
                f"There is no json file containing qubit "
                f"parameters in {experiment_directory}."
            )
        full_filepath = f"{experiment_directory}\\{full_filepath[0]}"
    with open(full_filepath) as f:
        qubit_parameters = f.read()
    # convert to python dictionary
    qubit_parameters = json.loads(qubit_parameters)
    return qubit_parameters


def load_qubits_from_experiment_directory(experiment_directory, measurement_setup):
    """
    Creates new instances of Transmon with the parameters loaded from a file
    in experiment_directory.

    Searched for filenames containing "qubit_parameters.yaml" or
    "qubit_parameters.json" in experiment_directory.

    Args:
        experiment_directory: path to the directory where the measurement data is saved
        measurement_setup: instance of DeviceSetup; passed to
            calib_hlp.create_qubits_from_parameters

    Returns:
        list of Transmon instances (see create_qubits_from_parameters)
    """

    try:
        qubit_parameters = load_qubit_parameters_from_yaml(
            experiment_directory + "\\qubit_parameters.yaml"
        )
    except FileNotFoundError:
        qubit_parameters = load_qubit_parameters_from_json(experiment_directory)
    qubits = calib_hlp.create_qubits_from_parameters(
        qubit_parameters, measurement_setup
    )
    measurement_setup.qubits = qubits
    return qubits
