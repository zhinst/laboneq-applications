import datetime
import time
import json
import dill as pickle
import os
from pathlib import Path

from laboneq.simple import *  # noqa: F403
from ruamel.yaml import YAML

ryaml = YAML()

import logging

log = logging.getLogger(__name__)


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


def save_results_to_database(
    results_database, results_object, key_name: str, user_note: str
):
    results_database.store(
        data=results_object,
        key=f"{key_name}_{datetime.datetime.now()}",
        metadata={
            "creation_date": datetime.datetime.now(),
            "user_note": f"{user_note}",
        },
    )


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
        if qubit in measurement_setup.qubits:
            qb_idx = measurement_setup.qubits.index(qubit)
            measurement_setup.qubits[qb_idx] = qubit
    measurement_setup.set_calibration(msmt_setup_cal)


def create_qubits_from_measurement_setup(measurement_setup):
    """
    Instantiates Transmons from the logical signals in measurement_setup and
    dome default values for the parameters.

    Args:
        measurement_setup: instance of DeviceSetup

    Returns:
        list of Transmon instances
    """

    qubits = []
    for qb_name in measurement_setup.logical_signal_groups:
        qubits += [
            Transmon.from_logical_signal_group(
                uid=qb_name,
                lsg=measurement_setup.logical_signal_groups[qb_name],
                parameters=TransmonParameters(
                    resonance_frequency_ge=6.00e9,
                    resonance_frequency_ef=5.83e9,
                    drive_lo_frequency=5.90e9,
                    readout_resonator_frequency=7.00e9,
                    readout_lo_frequency=7.30e9,
                    readout_integration_length=2e-06,
                    readout_integration_delay=2.4e-07,
                    drive_range=10,
                    readout_range_out=-25,
                    readout_range_in=-5,
                    flux_offset_voltage=0,
                    drive_parameters_ge={
                        "amplitude_pi": 0.30,
                        "amplitude_pi2": 0.15,
                        "beta": 0,
                        "length": 5e-08,
                        "sigma": 0.2,
                    },
                    drive_parameters_ef={
                        "amplitude_pi": 0.30,
                        "amplitude_pi2": 0.15,
                        "beta": 0,
                        "length": 5e-08,
                        "sigma": 0.2,
                    },
                    user_defined={
                        "dc_slot": 0,
                        "dc_voltage_parking": 0,
                        "readout_amplitude": 0.5,
                        "readout_length": 2e-06,
                        "reset_delay_length": 1e-06,
                        "spec_amplitude": 1,
                        "spec_length": 5e-06,
                    },
                ),
            )
        ]
    return qubits


def create_qubits_from_parameters(qubit_parameters, measurement_setup):
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
    for qb_name in qubit_parameters:
        transmon_parameters = TransmonParameters()
        for param_name, param_value in qubit_parameters[qb_name].items():
            setattr(transmon_parameters, param_name, param_value)
        qubits += [
            Transmon.from_logical_signal_group(
                qb_name,
                lsg=measurement_setup.logical_signal_groups[qb_name],
                parameters=transmon_parameters,
            )
        ]
    return qubits


def save_qubit_parameters(save_folder, qubits, timestamp=""):
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
    qb_pars_file = os.path.abspath(
        os.path.join(save_folder, f"{timestamp}_qubit_parameters.json")
    )
    with open(qb_pars_file, "w") as file:
        json.dump(qubit_parameters, file, indent=2)


class QubitTemporaryValuesContext:
    """
    This context manager allows to change a given qubit parameter
    to a new value, and the original value is reverted upon exit of the context
    manager.

    Args:
        *param_value_pairs: 3-tuples of qubit instance, qubits parameter name
            and its temporary value

    Example:
        # measure qubit spectroscopy at a different readout power without
        # setting the parameter value
        with QubitTemporaryValuesContext(
            (qb1, "readout_range_out", -5)
        ):
            ResonatorSpectroscopy(...)
    """

    def __init__(self, *param_value_pairs):
        if len(param_value_pairs) > 0 and not isinstance(
            param_value_pairs[0], (tuple, list)
        ):
            param_value_pairs = (param_value_pairs,)
        self.param_value_pairs = param_value_pairs
        self.old_value_pairs = []

    def __enter__(self):
        log.debug("Entered QubitTemporaryValuesContext")
        try:
            self.old_value_pairs = [
                (qubit, param_name, getattr(qubit.parameters, param_name))
                for qubit, param_name, _ in self.param_value_pairs
            ]
            for qubit, param_name, value in self.param_value_pairs:
                setattr(qubit.parameters, param_name, value)
        except KeyError as e:
            self.__exit__(None, None, None)
            raise KeyError(
                f"Trying to set temporary value to the qubit "
                f"parameter {e}, but this parameter does not "
                f"exist."
            )
        except Exception:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, type, value, traceback):
        for qubit, param_name, value in self.old_value_pairs:
            setattr(qubit.parameters, param_name, value)
        log.debug("Exited QubitTemporaryValuesContext")
