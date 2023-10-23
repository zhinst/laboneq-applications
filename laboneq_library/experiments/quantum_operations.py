from laboneq.simple import *  # noqa: F403


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
