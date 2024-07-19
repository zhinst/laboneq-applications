import functools

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


@functools.cache
def _cached_readout_pulse(length, amplitude):
    return pulse_library.const(
        length=length,
        amplitude=amplitude,
    )


def readout_pulse(qubit):
    return _cached_readout_pulse(
        length=qubit.parameters.readout_pulse_length,
        amplitude=qubit.parameters.readout_amplitude,
    )


@functools.cache
def _cached_kernel(length, amplitude):
    return pulse_library.const(
        length=length,
        amplitude=amplitude,
    )


def integration_kernel(qubit):
    return _cached_kernel(
        length=qubit.parameters.readout_pulse_length,
        amplitude=1.0,
    )


def drag(x, sigma=1 / 3, beta=0.2, zero_boundaries=False, **_):
    """
     Copy of the DRAG pulse in pulse_library but WITHOUT THE PULSE FUNCTIONAL
     DECORATOR IN ORDER TO BE ABLE TO EVALUATE IT!

    Arguments:
        **_ (Any):
            All pulses accept the following keyword arguments:
            - uid ([str][]): Unique identifier of the pulse
            - length ([float][]): Length of the pulse in seconds
            - amplitude ([float][]): Amplitude of the pulse
        sigma (float):
            Std. deviation, relative to pulse length
        beta (float):
            Relative amplitude of the quadrature component
        zero_boundaries (bool):
            Whether to zero the pulse at the boundaries

    Returns:
        array of the calculated envelope
    """
    gauss = np.exp(-(x**2) / (2 * sigma**2))
    delta = 0
    if zero_boundaries:
        dt = x[0] - (x[1] - x[0])
        delta = np.exp(-(dt**2) / (2 * sigma**2))
    d_gauss = -x / sigma**2 * gauss
    gauss -= delta
    return (gauss + 1j * beta * d_gauss) / (1 - delta)


@pulse_library.register_pulse_functional
def piecewise_modulated_drag(x, piece_pulse_parameters, piece_frequencies,
                             sampling_rate, **_):
    """
    Definition of a piecewise modulated waveform created from several
    concatenated drag pulses.

    Each drag pulse is created from the pulse parameters in piece_pulse_parameters,
    and is modulation at the corresponding frequency in piece_frequencies.
    """
    values = np.array([])
    time = 0.5 * (x+1) / sampling_rate

    for pulse_params, frequency in zip(
        piece_pulse_parameters, piece_frequencies
    ):
        if "uid" not in pulse_params:
            pulse_params["uid"] = None
        values = np.append(
            values,
            np.exp(-1j * 2 * np.pi * frequency * time) * drag(x, **pulse_params),
        )
    return values


def reset_pulse_ef(qubit):
    """
    Creates the pulse needed to perform active reset of the ef transition.

    This pulse is passed directly to the play command of the Case section.
    """
    piece_frequencies = [
        0,
        qubit.parameters.drive_frequency_ef - qubit.parameters.drive_frequency_ge
    ]
    piece_pulse_parameters = (qubit.parameters.drive_parameters_ge,
                              qubit.parameters.drive_parameters_ef)
    return piecewise_modulated_drag(
        uid=f'reset_pulse_ef_{qubit.uid}',
        piece_pulse_parameters=piece_pulse_parameters,
        piece_frequencies=piece_frequencies)
