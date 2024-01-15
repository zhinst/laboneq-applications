import numpy as np
from laboneq.simple import *  # noqa: F403


def drive_pulse(pulse_type, **pulse_kwargs):
    if "uid" not in pulse_kwargs:
        pulse_kwargs["uid"] = "drive_pulse"
    return pulse_type(**pulse_kwargs)


# GATES
def quantum_gates(qubit):
    ge_pars = qubit.parameters.drive_parameters_ge
    ef_pars = qubit.parameters.drive_parameters_ef
    gates = {}
    for suff, pars in zip(["ge", "ef"], [ge_pars, ef_pars]):
        gates.update(
            {
                f"X180_{suff}": dict(amplitude=pars["amplitude_pi"], phase=0),
                f"mX180_{suff}": dict(amplitude=-pars["amplitude_pi"], phase=0),
                f"X90_{suff}": dict(amplitude=pars["amplitude_pi2"], phase=0),
                f"mX90_{suff}": dict(amplitude=-pars["amplitude_pi2"], phase=0),
                f"Y180_{suff}": dict(amplitude=pars["amplitude_pi"], phase=np.pi / 2),
                f"mY180_{suff}": dict(amplitude=-pars["amplitude_pi"], phase=np.pi / 2),
                f"Y90_{suff}": dict(amplitude=pars["amplitude_pi2"], phase=np.pi / 2),
                f"mY90_{suff}": dict(amplitude=-pars["amplitude_pi2"], phase=np.pi / 2),
            }
        )
    return gates


def quantum_gate(
    qubit,
    gate_name,
    pulse_type=pulse_library.drag,
    uid=None,
    additional_pulse_parameters=None,
):
    pulse_pars = (
        qubit.parameters.drive_parameters_ef
        if "ef" in gate_name
        else qubit.parameters.drive_parameters_ge
    )
    gates = quantum_gates(qubit)
    if gate_name not in gates:
        raise KeyError(
            f"Gate {gate_name} is not defined. Available gates:\n"
            f"{list(gates.keys())}."
        )
    pulse_kwargs = {}
    pulse_kwargs.update(pulse_pars)
    pulse_kwargs.update(gates[gate_name])
    if additional_pulse_parameters is not None:
        pulse_kwargs.update(additional_pulse_parameters)
    if uid is None:
        uid = f"{gate_name} {qubit.uid}"
    return drive_pulse(pulse_type, uid=uid, **pulse_kwargs)
