from ruamel.yaml import YAML
from pathlib import Path
import time

from laboneq.simple import *

ryaml = YAML()


# saving and loading
def load_qubit_parameters(filename="./qubit_parameters.yaml"):
    calib_file = open(filename).read()
    qubit_parameters = ryaml.load(calib_file)
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
