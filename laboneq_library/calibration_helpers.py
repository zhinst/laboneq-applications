import datetime
import time
import json
import os
from pathlib import Path

from laboneq.simple import *  # noqa: F403
from ruamel.yaml import YAML

ryaml = YAML()


# saving and loading
def load_qubit_parameters(filename="./qubit_parameters.yaml"):
    with open(filename) as f:
        calib_file = f.read()
    qubit_parameters = ryaml.load(calib_file)
    return qubit_parameters


def load_qubit_parameters_json(folder=None, full_filepath=None):
    if full_filepath is None:
        if folder is None:
            raise ValueError('Please provide either folder or full_filepath.')
        full_filepath = [fn for fn in os.listdir(folder) if
                         'qubit_parameters.json' in fn]
        if len(full_filepath) == 0:
            raise FileNotFoundError(f'There is no json file containing qubit '
                                    f'parameters in {folder}.')
        full_filepath = f'{folder}\\{full_filepath[0]}'
    with open(full_filepath) as f:
        qubit_parameters = f.read()
    # convert to python dictionary
    qubit_parameters = json.loads(qubit_parameters)
    return qubit_parameters


def update_qubit_parameters_and_calibration(
    qubit_parameters,
    device_setup,
    database=None,
    calibration_file="./qubit_parameters.yaml",
    history_path="calib_history",
    set_local_oscillators=True,
):
    with open(calibration_file, "w") as file:
        ryaml.dump(qubit_parameters, file)

    Path(history_path).mkdir(parents=True, exist_ok=True)

    timestamp = str(time.strftime("%Y-%m-%d_%H%M%S"))
    history_filename = history_path + "/" + timestamp + "calib.yaml"
    with open(history_filename, "w") as file:
        ryaml.dump(qubit_parameters, file)

    transmon_list = []
    for qubit in qubit_parameters["qubits"]:
        transmon = create_transmon(qubit, qubit_parameters, device_setup)
        device_setup.set_calibration(
            transmon.calibration(set_local_oscillators=set_local_oscillators)
        )
        transmon_list.append(transmon)

    if database is not None:
        database.store(
            data=device_setup,
            key=str(datetime.datetime.now()),
            metadata={"creation_date": datetime.datetime.now(), "name": "device_setup"},
        )

        database.store(
            data=device_setup.get_calibration(),
            key=str(datetime.datetime.now()),
            metadata={"creation_date": datetime.datetime.now(), "name": "calibration"},
        )
    return transmon_list


def update_measurement_setup_from_qubits(qubits, measurement_setup):
    """
    Overwrites measurement_setup.qubit with qubits, and the measurement-setup
    calibration with the qubits calibration.

    Args:
        qubits: list of qubits
        measurement_setup: instance of DeviceSetup
    """

    msmt_setup_cal = measurement_setup.get_calibration()
    for qubit in qubits:
        cal = qubit.calibration()
        for sig_name, lsig_path in qubit.signals.items():
            msmt_setup_cal[lsig_path] = cal[lsig_path]
    measurement_setup.set_calibration(msmt_setup_cal)
    measurement_setup.qubits = qubits


# create a transmon qubit object from entries in a parameter dictionary
def create_transmon(qubit: str, base_parameters, device_setup):
    q_name = qubit
    transmon = Transmon.from_logical_signal_group(
        q_name,
        lsg=device_setup.logical_signal_groups[q_name],
        parameters=TransmonParameters(
            resonance_frequency_ge=base_parameters["qubits"][qubit][
                "resonance_frequency_ge"
            ]["value"],
            resonance_frequency_ef=base_parameters["qubits"][qubit][
                "resonance_frequency_ef"
            ]["value"],
            drive_lo_frequency=base_parameters["qubits"][qubit]["drive_lo_frequency"][
                "value"
            ],
            readout_resonator_frequency=base_parameters["qubits"][qubit][
                "readout_resonator_frequency"
            ]["value"],
            readout_lo_frequency=base_parameters["qubits"][qubit][
                "readout_lo_frequency"
            ]["value"],
            readout_integration_delay=base_parameters["qubits"][qubit][
                "readout_integration_delay"
            ]["value"],
            drive_range=base_parameters["qubits"][qubit]["drive_range_ge"]["value"],
            readout_range_out=base_parameters["qubits"][qubit]["readout_range_out"][
                "value"
            ],
            readout_range_in=base_parameters["qubits"][qubit]["readout_range_in"][
                "value"
            ],
            flux_offset_voltage=base_parameters["qubits"][qubit]["dc_source"]["value"],
            user_defined={
                "amplitude_pi": base_parameters["qubits"][qubit]["amplitude_pi"][
                    "value"
                ],
                "amplitude_pi2": base_parameters["qubits"][qubit]["amplitude_pi2"][
                    "value"
                ],
                "amplitude_pi_ef": base_parameters["qubits"][qubit]["amplitude_pi_ef"][
                    "value"
                ],
                "amplitude_pi2_ef": base_parameters["qubits"][qubit][
                    "amplitude_pi2_ef"
                ]["value"],
                "drive_range_ef": base_parameters["qubits"][qubit]["drive_range_ef"][
                    "value"
                ],
                "pulse_length": base_parameters["qubits"][qubit]["pulse_length"][
                    "value"
                ],
                "readout_length": base_parameters["qubits"][qubit]["readout_length"][
                    "value"
                ],
                "readout_amplitude": base_parameters["qubits"][qubit][
                    "readout_amplitude"
                ]["value"],
                "reset_delay_length": base_parameters["qubits"][qubit][
                    "reset_delay_length"
                ]["value"],
                "dc_slot": base_parameters["qubits"][qubit]["dc_source"]["slot"],
                "cr_frequency": base_parameters["qubits"][qubit]["cr_freq"]["value"],
            },
        ),
    )
    return transmon


def save_results(results_database, results_object, key_name: str, user_note: str):
    results_database.store(
        data=results_object,
        key=f"{key_name}_{datetime.datetime.now()}",
        metadata={
            "creation_date": datetime.datetime.now(),
            "user_note": f"{user_note}",
        },
    )


def create_qubits(qubit_parameters, measurement_setup):
    """
    Instantiates Transmons from the logical signals in measurement_setup and
    the qubit_parameters.

    Args:
        qubit_parameters: dictionary containing qubit parameters expected by
            TransmonParameters. Has the form:
            {qubit_name: {parameter_name: parameter_value}}
        measurement_setup: instance of DeviceSetup

    Returns:
        list of Transmon instances
    """

    qubits = []
    parameters = qubit_parameters
    for q_name in qubit_parameters:
        qubits += [Transmon.from_logical_signal_group(
            q_name,
            lsg=measurement_setup.logical_signal_groups[q_name],
            parameters=TransmonParameters(
                resonance_frequency_ge=parameters[q_name][
                    "resonance_frequency_ge"],
                resonance_frequency_ef=parameters[q_name][
                    "resonance_frequency_ef"],
                drive_lo_frequency=parameters[q_name]["drive_lo_frequency"],
                readout_resonator_frequency=parameters[q_name][
                    "readout_resonator_frequency"],
                readout_lo_frequency=parameters[q_name][
                    "readout_lo_frequency"],
                readout_integration_delay=parameters[q_name][
                    "readout_integration_delay"],
                drive_range=parameters[q_name]["drive_range"],
                drive_parameters_ge=parameters[q_name]["drive_parameters_ge"],
                drive_parameters_ef=parameters[q_name]["drive_parameters_ef"],
                readout_range_out=parameters[q_name]["readout_range_out"],
                readout_range_in=parameters[q_name]["readout_range_in"],
                flux_offset_voltage=parameters[q_name]["flux_offset_voltage"],
                user_defined=parameters[q_name]['user_defined'],
            ),
        )]
    return qubits


def load_measurement_setup_from_data_folder(data_folder):
    """
    Load a DeviceSetup from the data_folder.

    Searched for a filename that contains "measurement_setup.json" inside
    data_folder.

    Args:
        data_folder: path to the directory where the measurement data is saved

    Returns:
        instance of DeviceSetup
    """

    msmt_setup_fn = [f for f in os.listdir(data_folder)
                     if "measurement_setup.json" in f]
    if len(msmt_setup_fn) == 0:
        raise ValueError(f"The data folder {data_folder} does not contain a "
                         f"measurement_setup.json file.")
    else:
        msmt_setup_fn = msmt_setup_fn[0]
    return DeviceSetup.load(data_folder + f'\\{msmt_setup_fn}')


def load_qubits_from_data_folder(data_folder, measurement_setup):
    """
    Creates new instances of Transmon with the parameters loaded from a file
    in data_folder.

    Searched for filenames containing "qubit_parameters.yaml" or
    "qubit_parameters.json" in data_folder.

    Args:
        data_folder: path to the directory where the measurement data is saved
        measurement_setup: instance of DeviceSetup; passed to create_qubits

    Returns:
        list of Transmon instances (see create_qubits)
    """

    try:
        qubit_parameters = load_qubit_parameters(
            data_folder + '\\qubit_parameters.yaml')
    except FileNotFoundError:
        qubit_parameters = load_qubit_parameters_json(data_folder)
    qubits = create_qubits(qubit_parameters, measurement_setup)
    measurement_setup.qubits = qubits
    return qubits


def save_qubit_parameters(save_folder, qubits, timestamp=''):
    """
    Saves the parameters of qubits into a json file.

    Args:
        save_folder: folder in which to save the json file
        qubits: list of qubit instances
        timestamp: string with the timestamp with the format "YYYYMMDD_hhmmss."
            The timestamp is prepended to the json filename.
    """

    qubit_parameters = {qb.uid: qb.parameters.__dict__ for qb in qubits}
    # Save all qubit parameters in one json file
    qb_pars_file = os.path.abspath(os.path.join(
        save_folder, f'{timestamp}_qubit_parameters.json'))
    with open(qb_pars_file, "w") as file:
        json.dump(qubit_parameters, file, indent=2)


def get_latest_data_folder(data_directory):
    """
    Returns the last data folder in data_directory.

    Example: data_directory contains the following folders:
    ["20231128", "20231127", "20231126"]. This function returns the most recent
    folder inside data_directory/20231128, as sorted by the timestamp in the
    names of the folders. 

    Args:
        data_directory: directory where the measurement data is saved

    Returns:
        the latest data folder
    """
    day_folders = os.listdir(data_directory)
    day_folders.sort()
    day_folder = day_folders[-1]
    ts_folders = os.listdir(data_directory + f'\\{day_folder}')
    ts_folders.sort()
    ts_folder = ts_folders[-1]
    latest_folder = data_directory + f'\\{day_folder}\\{ts_folder}'
    return latest_folder
