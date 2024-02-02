import json
import os
import logging
from laboneq.simple import *  # noqa: F403
from laboneq_library.qpu_types.tunable_transmon import (
    TunableTransmonQubit,
    TunableTransmonQubitParameters,
)


log = logging.getLogger(__name__)


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


def create_qubits_from_parameters(qubit_parameters, measurement_setup):
    """
    Instantiates TunableTransmonQubits from the logical signals in measurement_setup and
    the qubit_parameters.

    Args:
        qubit_parameters: dictionary containing qubit parameters expected by
            TunableTransmonQubitParameters. Has the form:
            {qubit_name: {parameter_name: parameter_value}}
        measurement_setup: instance of DeviceSetup

    Returns:
        list of TunableTransmonQubit instances
    """

    qubits = []
    for qb_name in qubit_parameters:
        transmon_parameters = TunableTransmonQubitParameters(
            **qubit_parameters[qb_name]
        )
        qubits += [
            TunableTransmonQubit.from_logical_signal_group(
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
