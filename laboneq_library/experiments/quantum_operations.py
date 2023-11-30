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
    qubit, gate_name, pulse_type=pulse_library.drag, uid=None,
    additional_pulse_parameters=None
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


def drive_ge_pi(qubit, amplitude=None):
    return pulse_library.gaussian(
        length=qubit.parameters.user_defined["pulse_length"],
        amplitude=amplitude
        if amplitude is not None
        else qubit.parameters.user_defined["amplitude_pi"],
    )


def drive_ge_pi2(qubit, amplitude=None):
    return pulse_library.gaussian(
        length=qubit.parameters.user_defined["pulse_length"],
        amplitude=amplitude
        if amplitude is not None
        else qubit.parameters.user_defined["amplitude_pi2"],
    )


def readout_pulse(qubit):
    return pulse_library.const(
        length=qubit.parameters.user_defined["readout_length"],
        amplitude=qubit.parameters.user_defined["readout_amplitude"],
    )


def integration_kernel(qubit):
    return pulse_library.const(
        length=qubit.parameters.user_defined["readout_length"],
        amplitude=1,
    )


### Legacy below this line
"""

# pulses
def qubit_gaussian_pulse(qubit):
    return pulse_library.gaussian(
        uid=f"gaussian_pulse_drive_{qubit.uid}",
        length=qubit.parameters.user_defined["pulse_length"],
        amplitude = 1.0,
        sigma=0.3,
    )

def qubit_drive_pulse(qubit, amplitude=None):
    if amplitude is None:
        amplitude = qubit.parameters.user_defined["amplitude_pi"]
    return pulse_library.drag(
        uid=f"drag_pulse_{qubit.uid}",
        length=qubit.parameters.user_defined["pulse_length"],
        amplitude=amplitude,
        sigma=0.3,
        beta=0.0,
    )

def readout_gauss_square_pulse(qubit):
    return pulse_library.gaussian_square(
        uid=f"readout_pulse_{qubit.uid}",
        length=qubit.parameters.user_defined["readout_length"],
        amplitude=qubit.parameters.user_defined["readout_amplitude"],
    )

def qubit_spectroscopy_pulse(qubit):
    return pulse_library.const(
        uid=f"spectroscopy_pulse_{qubit.uid}",
        length=qubit.parameters.user_defined["readout_length"],
        amplitude=qubit.parameters.user_defined["readout_amplitude"],
        # can_compress=True,
    )

def qubit_gaussian_pulse(qubit):
    return pulse_library.gaussian(
        uid=f"gaussian_pulse_drive_{qubit.uid}",
        length=qubit.parameters.user_defined["pulse_length"],
        amplitude = qubit.parameters.user_defined["amplitude_pi"],
    )

def qubit_gaussian_halfpi_pulse(qubit):
    return pulse_library.gaussian(
        uid=f"gaussian_pulse_drive_{qubit.uid}",
        length=qubit.parameters.user_defined["pulse_length"],
        amplitude = qubit.parameters.user_defined["amplitude_pi2"],
    )

"""
