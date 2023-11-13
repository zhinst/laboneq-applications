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


def update_setup_calibration_from_qubits(qubits, measurement_setup):
    msmt_setup_cal = measurement_setup.get_calibration()
    for qubit in qubits:
        for sig_name, lsig_path in qubit.signals.items():
            msmt_setup_cal[lsig_path] = qubit.calibration()[lsig_path]
    measurement_setup.set_calibration(msmt_setup_cal)


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


def reload_qubit_parameters(folder, measurement_setup):
    try:
        qubit_parameters = load_qubit_parameters(folder + '\\qubit_parameters.yaml')
    except FileNotFoundError:
        qubit_parameters = load_qubit_parameters_json(folder)
    return create_qubits(qubit_parameters, measurement_setup)


def save_qubit_parameters(savedir, qubits, timestamp=''):
    qubit_parameters = {qb.uid: qb.parameters.__dict__ for qb in qubits}
    # Save all qubit parameters in one yaml file
    qb_pars_file = os.path.abspath(os.path.join(
        savedir, f'{timestamp}_qubit_parameters.json'))
    with open(qb_pars_file, "w") as file:
        json.dump(qubit_parameters, file, indent=2)


def fit_data_lmfit(function, x, y, param_hints):
    import lmfit
    model = lmfit.Model(function)
    model.param_hints = param_hints
    return model.fit(x=x, data=y, params=model.make_params())


def flatten_lmfit_modelresult(fit_result):
    import lmfit
    # used for saving an lmfit ModelResults object as a dict
    assert type(fit_result) is lmfit.model.ModelResult
    fit_res_dict = dict()
    fit_res_dict['success'] = fit_result.success
    fit_res_dict['message'] = fit_result.message
    fit_res_dict['params'] = {}
    for param_name in fit_result.params:
        fit_res_dict['params'][param_name] = {}
        param = fit_result.params[param_name]
        for k in param.__dict__:
            if k == '_val':
                fit_res_dict['params'][param_name]['value'] = getattr(param, k)
            else:
                if not k.startswith('_') and k not in ['from_internal', ]:
                    fit_res_dict['params'][param_name][k] = getattr(param, k)
    return fit_res_dict