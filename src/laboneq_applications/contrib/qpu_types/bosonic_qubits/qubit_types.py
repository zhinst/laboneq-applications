# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Bosonic qubits and parameters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import attrs
import numpy as np
from laboneq.core.types.enums.port_mode import PortMode
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.calibration import Calibration, Oscillator, SignalCalibration
from laboneq.dsl.enums import ModulationType
from laboneq.dsl.quantum import (
    QuantumElement,
    QuantumParameters,
)
from laboneq.simple import dsl

if TYPE_CHECKING:
    from laboneq.dsl.experiment.pulse import Pulse

# Maximum number of photon-number-selective signal lines
# that can be defined. The actual number used is controlled
# by the `max_photon_number` qubit parameter.
_MAX_SUPPORTED_PHOTON_NUMBER = 10


@classformatter
@attrs.define(kw_only=True)
class BosonicQubitParameters(QuantumParameters):
    """Qubit parameters for `BosonicQubit` instances.

    A bosonic qubit encodes quantum information in a high-Q memory
    cavity, controlled via a dispersively coupled ancilla transmon,
    and read out through a dedicated readout resonator.

    Due to the dispersive interaction (chi), the transmon frequency
    depends on the photon number in the memory:

        f_transmon(n) = f_transmon(0) + n * chi

    All transmon drive parameters are organized around this
    number-dependent structure. The parameters for n=0 also serve
    as the default parameters for standard transmon operations
    (x90, x180, Ramsey, etc.).

    Attributes:
        # --- Memory cavity parameters ---

        memory_lo_frequency:
            Local oscillator frequency for the memory cavity drive.
        memory_resonance_frequency:
            Resonance frequency of the memory cavity.
        memory_T1:
            Energy relaxation time of the memory cavity.
        memory_T2:
            Coherence time (echo) of the memory cavity.
        memory_T2_star:
            Coherence time (Ramsey) of the memory cavity.
        memory_self_kerr:
            Self-Kerr nonlinearity of the memory cavity (Hz).
        memory_drive_amplitude:
            Default drive amplitude for the memory cavity.
        memory_drive_length:
            Default drive pulse length for the memory cavity (seconds).
        memory_drive_pulse:
            Pulse parameters for memory cavity drive pulses.
        memory_drive_range:
            Power range for the memory cavity drive line (dBm).
        memory_spectroscopy_amplitude:
            Amplitude of the memory cavity spectroscopy pulse.
        memory_spectroscopy_length:
            Length of the memory cavity spectroscopy pulse (seconds).
        memory_spectroscopy_pulse:
            Pulse parameters for memory cavity spectroscopy pulses.

        # --- Ancilla transmon parameters ---

        transmon_lo_frequency:
            Local oscillator frequency for the ancilla transmon drive.
        transmon_T1:
            Energy relaxation time of the ancilla transmon.
        transmon_T2:
            Coherence time (echo) of the ancilla transmon.
        transmon_T2_star:
            Coherence time (Ramsey) of the ancilla transmon.
        transmon_resonance_frequency_at_n0:
            Resonance frequency of the ancilla transmon when the memory
            cavity is in the vacuum state (|0⟩). This is the base
            frequency from which all number-dependent frequencies
            are derived: f(n) = f(0) + n * chi.
        transmon_drive_range:
            Power range for the ancilla transmon drive line (dBm).
        transmon_spectroscopy_amplitude:
            Amplitude of the transmon spectroscopy pulse.
        transmon_spectroscopy_length:
            Length of the transmon spectroscopy pulse (seconds).
        transmon_spectroscopy_pulse:
            Pulse parameters for transmon spectroscopy pulses.

        # --- Selective transmon Rx drive parameters ---

        max_photon_number:
            Maximum number of cavity Fock states to consider for
            number-selective operations (SNAP gate, parity
            measurement, Wigner tomography). Determines how many
            selective drive signal lines are calibrated.
            Default: 5.
        selective_transmon_Rx_length:
            Length of the selective transmon drive pulse (seconds).
            Shared across all photon numbers. Must satisfy
            selective_transmon_Rx_length >> 1/|chi| for frequency
            selectivity.
            Default: 2 µs.
        selective_transmon_Rx_pulse:
            Pulse parameters for the selective transmon drive.
            Shared across all photon numbers. A Gaussian pulse is
            recommended for minimal spectral leakage.
        selective_transmon_X180_amplitudes:
            List of calibrated π-pulse amplitudes for each photon
            number n = 0, 1, ..., max_photon_number.
            Index 0 is the standard transmon π amplitude (when the
            memory is in |0⟩).
            Must have length >= max_photon_number + 1.
        selective_transmon_X90_amplitudes:
            List of calibrated π/2-pulse amplitudes for each photon
            number n = 0, 1, ..., max_photon_number.
            Index 0 is the standard transmon π/2 amplitude.
            Must have length >= max_photon_number + 1.

        # --- SNAP gate parameters ---

        snap_drive_amp_per_n:
            Dictionary mapping photon number n (as str) to the calibrated
            selective π-pulse amplitude for the SNAP gate.
        snap_drive_phase_per_n:
            Dictionary mapping photon number n (as str) to the calibrated
            drive-phase offset for the SNAP gate (radians).

        # --- SWAP gate parameters ---

        swap_length:
            Calibrated SWAP pulse duration (seconds).
        swap_amplitude:
            Calibrated SWAP pulse amplitude.
        swap_pulse:
            Envelope definition for the SWAP pulse. A constant
            (square) pulse is the default.
        swap_drive_range:
            Voltage range for the SWAP drive line (dBm).
            Default: 5.

        # --- Cross-coupling parameters ---

        chi:
            Dispersive shift (cross-Kerr) between memory cavity
            and ancilla transmon (Hz).
            Convention: f_transmon(n) = f_transmon(0) + n * chi.

        # --- Readout parameters ---

        readout_lo_frequency:
            Local oscillator frequency for the readout line.
        readout_resonator_frequency:
            Readout resonator frequency.
        readout_amplitude:
            Readout pulse amplitude.
        readout_length:
            Readout pulse length (seconds).
        readout_pulse:
            Pulse parameters for the readout pulse.
        readout_integration_length:
            Duration of the weighted integration (seconds).
        readout_integration_delay:
            Integration delay between readout pulse and data
            acquisition (seconds). Defaults to 20 ns.
        readout_integration_kernels_type:
            The type of integration kernel: "default" or "optimal".
        readout_integration_kernels:
            Either "default" or a list of pulse dictionaries.
        readout_integration_discrimination_thresholds:
            Either None or a list of thresholds.
        readout_range_out:
            Readout output power range (dBm).
        readout_range_in:
            Readout input power range (dBm).

        # --- Reset parameters ---

        reset_delay_length:
            Duration of the wait time for passive reset (seconds).

        # --- Displacement calibration result ---

        displacement_amp_per_unit_beta:
            Calibrated scale factor mapping drive amplitude to
            phase-space displacement: amp_hardware =
            displacement_amp_per_unit_beta * |beta|, This is the
            slope of the linear fit from the displacement
            calibration experiment.

        # --- Flux parameters ---

        flux_range:
            Voltage range for the flux control line (volts).
        dc_slot:
            Slot number on the DC source for applying DC voltage.
        dc_voltage_parking:
            DC parking voltage.
    """

    # ----------------------------------------------------------------
    # Memory cavity parameters
    # ----------------------------------------------------------------

    memory_T1: float = 0  # noqa: N815
    memory_T2: float = 0  # noqa: N815
    memory_T2_star: float = 0  # noqa: N815
    memory_self_kerr: float = 0

    memory_lo_frequency: float | None = None
    memory_resonance_frequency: float | None = None

    memory_drive_amplitude: float = 0.2
    memory_drive_length: float = 100e-9
    memory_drive_pulse: dict = attrs.field(
        factory=lambda: {
            "function": "const",
        },
    )
    memory_drive_range: float = 10

    memory_spectroscopy_amplitude: float | None = 1
    memory_spectroscopy_length: float | None = 5e-6
    memory_spectroscopy_pulse: dict = attrs.field(
        factory=lambda: {
            "function": "const",
            "can_compress": True,
        },
    )

    # ----------------------------------------------------------------
    # Ancilla transmon parameters
    # ----------------------------------------------------------------

    transmon_T1: float = 0  # noqa: N815
    transmon_T2: float = 0  # noqa: N815
    transmon_T2_star: float = 0  # noqa: N815

    transmon_lo_frequency: float | None = None
    transmon_resonance_frequency_at_n0: float | None = None
    transmon_drive_range: float = 10

    transmon_spectroscopy_amplitude: float | None = 1
    transmon_spectroscopy_length: float | None = 5e-6
    transmon_spectroscopy_pulse: dict = attrs.field(
        factory=lambda: {
            "function": "const",
            "can_compress": True,
        },
    )

    # ----------------------------------------------------------------
    # selective transmon Rx drive parameters
    # ----------------------------------------------------------------

    max_photon_number: int = 5

    selective_transmon_Rx_length: float = 2e-6  # noqa: N815
    selective_transmon_Rx_pulse: dict = attrs.field(  # noqa: N815
        factory=lambda: {
            "function": "gaussian",
            "sigma": 0.25,
        },
    )
    selective_transmon_X180_amplitudes: list[float] = attrs.field(  # noqa: N815
        factory=lambda: [0.1] * 6,
    )
    selective_transmon_X90_amplitudes: list[float] = attrs.field(  # noqa: N815
        factory=lambda: [0.05] * 6,
    )

    # ----------------------------------------------------------------
    # SNAP gate parameters
    # ----------------------------------------------------------------
    # Both are dictionaries keyed by Fock-state index (as str),
    # e.g. {"0": 0.45, "1": 0.42, "2": 0.39, ...}

    snap_drive_amp_per_n: dict[str, float] = attrs.field(factory=dict)
    snap_drive_phase_per_n: dict[str, float] = attrs.field(factory=dict)

    # ----------------------------------------------------------------
    # SWAP gate parameters
    # ----------------------------------------------------------------
    # Used by the transmon-π + SWAP sequence to prepare Fock
    # state |1⟩ in the memory cavity via |g,0⟩ → |e,0⟩ → |g,1⟩.
    # The SWAP is driven as a DC flux pulse on a dedicated SG
    # channel configured in LF mode.

    swap_length: float = 500e-9
    swap_amplitude: float = 0.1
    swap_pulse: dict = attrs.field(
        factory=lambda: {
            "function": "const",
        },
    )
    swap_drive_range: float = 5

    # ----------------------------------------------------------------
    # Memory-Transmon-coupling parameters
    # ----------------------------------------------------------------

    chi: float = 0

    # ----------------------------------------------------------------
    # Readout parameters
    # ----------------------------------------------------------------

    readout_lo_frequency: float | None = None
    readout_resonator_frequency: float | None = None

    readout_amplitude: float = 1.0
    readout_length: float = 2e-6
    readout_pulse: dict = attrs.field(
        factory=lambda: {
            "function": "const",
        },
    )
    readout_integration_length: float = 2e-6
    readout_integration_delay: float = 20e-9
    readout_integration_kernels_type: Literal["default", "optimal"] = "default"
    readout_integration_kernels: list[dict] | None = None
    readout_integration_discrimination_thresholds: list[float] | None = None

    readout_range_out: float = 5
    readout_range_in: float = 10

    # ----------------------------------------------------------------
    # Displacement calibration result
    # ----------------------------------------------------------------

    displacement_amp_per_unit_beta: float | None = None

    # ----------------------------------------------------------------
    # Reset parameters
    # ----------------------------------------------------------------

    reset_delay_length: float | None = 1e-6

    # ----------------------------------------------------------------
    # Flux parameters
    # ----------------------------------------------------------------

    flux_range: float = 5
    dc_slot: int | None = 0
    dc_voltage_parking: float | None = 0.0

    # ----------------------------------------------------------------
    # Validation
    # ----------------------------------------------------------------

    def __attrs_post_init__(self) -> None:
        """Validate that amplitude lists match max_photon_number."""
        n_needed = self.max_photon_number + 1
        if len(self.selective_transmon_X180_amplitudes) < n_needed:
            raise ValueError(
                f"selective_transmon_X180_amplitudes has length "
                f"{len(self.selective_transmon_X180_amplitudes)} but "
                f"max_photon_number={self.max_photon_number} "
                f"requires at least {n_needed} entries."
            )
        if len(self.selective_transmon_X90_amplitudes) < n_needed:
            raise ValueError(
                f"selective_transmon_X90_amplitudes has length "
                f"{len(self.selective_transmon_X90_amplitudes)} but "
                f"max_photon_number={self.max_photon_number} "
                f"requires at least {n_needed} entries."
            )

    # ----------------------------------------------------------------
    # helper methods
    # ----------------------------------------------------------------

    @property
    def memory_drive_frequency(self) -> float | None:
        """Memory cavity drive baseband frequency."""
        if self.memory_lo_frequency is None or self.memory_resonance_frequency is None:
            return None
        return self.memory_resonance_frequency - self.memory_lo_frequency

    @property
    def readout_frequency(self) -> float | None:
        """Readout baseband frequency."""
        if (
            self.readout_lo_frequency is None
            or self.readout_resonator_frequency is None
        ):
            return None
        return self.readout_resonator_frequency - self.readout_lo_frequency

    def transmon_resonance_frequency(self, n: int = 0) -> float | None:
        """Transmon resonance frequency conditioned on photon number n.

        f_transmon(n) = f_transmon(0) + n * chi

        Arguments:
            n: The cavity photon number. Default: 0.

        Returns:
            The RF frequency, or None if the base frequency is not set.

        Raises:
            ValueError:
                If n is negative or exceeds max_photon_number.
        """
        if n < 0 or n > self.max_photon_number:
            raise ValueError(
                f"Photon number n={n} is out of range [0, {self.max_photon_number}]."
            )
        f0 = self.transmon_resonance_frequency_at_n0
        if f0 is None:
            return None
        return f0 + n * self.chi

    def transmon_drive_frequency(self, n: int = 0) -> float | None:
        """Transmon baseband drive frequency for photon number n.

        Arguments:
            n: The cavity photon number. Default: 0.

        Returns:
            The baseband frequency, or None if required frequencies
            are not set.

        Raises:
            ValueError:
                If n is negative or exceeds max_photon_number.
        """
        f_rf = self.transmon_resonance_frequency(n)
        if f_rf is None or self.transmon_lo_frequency is None:
            return None
        return f_rf - self.transmon_lo_frequency

    def selective_transmon_X180_amplitude(self, n: int = 0) -> float:  # noqa: N802
        """Calibrated selective π-pulse amplitude for photon number n.

        Arguments:
            n: The cavity photon number. Default: 0.

        Returns:
            The selective π-pulse amplitude.

        Raises:
            ValueError:
                If n is negative or exceeds max_photon_number.
        """
        if n < 0 or n > self.max_photon_number:
            raise ValueError(
                f"Photon number n={n} is out of range [0, {self.max_photon_number}]."
            )
        return self.selective_transmon_X180_amplitudes[n]

    def selective_transmon_X90_amplitude(self, n: int = 0) -> float:  # noqa: N802
        """Calibrated selective π/2-pulse amplitude for photon number n.

        Arguments:
            n: The cavity photon number. Default: 0.

        Returns:
            The selective π/2-pulse amplitude.

        Raises:
            ValueError:
                If n is negative or exceeds max_photon_number.
        """
        if n < 0 or n > self.max_photon_number:
            raise ValueError(
                f"Photon number n={n} is out of range [0, {self.max_photon_number}]."
            )
        return self.selective_transmon_X90_amplitudes[n]

    def preparation_amplitude_for_photon_number(
        self,
        n: int,
    ) -> float:
        """Compute the displacement amplitude to prepare ⟨n̂⟩ ≈ n.

        For a coherent state |alpha⟩, the mean photon number is
        n̄ = |beta|². To prepare a state with n̄ ≈ n, we need
        |beta| = sqrt(n), so:

            amp = displacement_amp_per_unit_beta * np.sqrt(n)

        Arguments:
            n: Target mean photon number.

        Returns:
            The calibrated drive amplitude.

        Raises:
            ValueError:
                If displacement_amp_per_unit_beta is not
                calibrated.
        """
        if self.displacement_amp_per_unit_beta is None:
            raise ValueError(
                "displacement_amp_per_unit_beta is not calibrated. "
                "Run the displacement calibration experiment first, "
                "or provide preparation_amplitudes explicitly."
            )
        return self.displacement_amp_per_unit_beta * np.sqrt(n)


@classformatter
@attrs.define
class BosonicQubit(QuantumElement):
    """A bosonic qubit encoded in a memory cavity with a transmon ancilla.

    The bosonic qubit consists of three modes:
    - **Memory cavity**: stores the bosonic-encoded quantum state.
    - **Ancilla transmon**: provides nonlinearity for control and
      readout of the memory cavity via the dispersive interaction.
    - **Readout resonator**: used for dispersive readout of the
      ancilla transmon state.

    All transmon drive operations are parameterized by photon number n.
    Standard operations (x90, x180, Ramsey) default to n=0.
    Number-selective operations (SNAP gate) use per-n signal lines.

    Signal lines:
        Required:
            - `drive_memory`: drive line for the memory cavity.
            - `drive_transmon`: standard drive for the transmon (n=0).
            - `measure`: readout pulse output.
            - `acquire`: readout acquisition.
        Optional:
            - `swap_drive`: dedicated LF line (second SG channel)
              used for DC flux SWAP pulses between the ancilla
              transmon and the memory cavity.
            - `drive_transmon_selective_n0` ...
              `drive_transmon_selective_n9`:
              Number-selective transmon drive lines, each tuned to
              f_transmon(n) = f_transmon(0) + n * chi.
              These are mapped to the same physical output as
              `drive_transmon` but have independent oscillators.
    """

    PARAMETERS_TYPE = BosonicQubitParameters
    REQUIRED_SIGNALS = (
        "drive_memory",
        "acquire",
        "measure",
    )
    OPTIONAL_SIGNALS = (
        "drive_transmon",
        "swap_drive",
        *(f"drive_transmon_at_n{n}" for n in range(_MAX_SUPPORTED_PHOTON_NUMBER + 1)),
    )

    @staticmethod
    def selective_signal_name(n: int) -> str:
        """Return the signal name for photon-number-selective drive.

        Arguments:
            n: The cavity photon number.

        Returns:
            The signal name, e.g. "drive_transmon_at_n2".
        """
        return f"drive_transmon_at_n{n}"

    def selective_transmon_Rx_parameters(  # noqa: N802
        self,
        n: int = 0,
    ) -> tuple[str, dict]:
        """Return the transmon drive line and parameters for photon number n.

        The drive parameters are photon number-dependent due to the dispersive
        interaction: f_transmon(n) = f_transmon(0) + n * chi.

        For n=0, returns the standard `drive_transmon_at_n0` signal line.
        For n>0, returns the selective `drive_transmon_at_nX` line.

        Arguments:
            n: The cavity photon number. Default: 0.

        Returns:
            line:
                The signal line name.
            params:
                Dictionary with keys: amplitude_pi, amplitude_pi2,
                length, pulse.
        """
        line = self.selective_signal_name(n)
        return line, {
            "amplitude_pi": (self.parameters.selective_transmon_X180_amplitude(n)),
            "amplitude_pi2": (self.parameters.selective_transmon_X90_amplitude(n)),
            "length": self.parameters.selective_transmon_Rx_length,
            "pulse": self.parameters.selective_transmon_Rx_pulse,
        }

    def memory_drive_parameters(self) -> tuple[str, dict]:
        """Return the memory cavity drive line and drive parameters.

        Returns:
            line:
                The memory cavity drive line.
            params:
                The memory cavity drive parameters.
        """
        param_keys = ["amplitude", "length", "pulse"]
        params = {k: getattr(self.parameters, f"memory_drive_{k}") for k in param_keys}
        return "drive_memory", params

    def readout_parameters(self) -> tuple[str, dict]:
        """Return the measure line and the readout parameters.

        Returns:
            line:
                The measure line of the qubit.
            params:
                The readout parameters.
        """
        param_keys = ["amplitude", "length", "pulse"]
        params = {k: getattr(self.parameters, f"readout_{k}") for k in param_keys}
        return "measure", params

    def readout_integration_parameters(self) -> tuple[str, dict]:
        """Return the acquire line and readout integration parameters.

        Returns:
            line:
                The acquire line of the qubit.
            params:
                The readout integration parameters.
        """
        param_keys = [
            "length",
            "kernels",
            "kernels_type",
            "discrimination_thresholds",
        ]
        params = {
            k: getattr(self.parameters, f"readout_integration_{k}") for k in param_keys
        }
        return "acquire", params

    def memory_spectroscopy_parameters(self) -> tuple[str, dict]:
        """Return the memory spectroscopy line and pulse parameters.

        Returns:
            line:
                The memory cavity drive line.
            params:
                The memory spectroscopy pulse parameters.
        """
        param_keys = ["amplitude", "length", "pulse"]
        params = {
            k: getattr(self.parameters, f"memory_spectroscopy_{k}") for k in param_keys
        }
        return "drive_memory", params

    def transmon_spectroscopy_parameters(self) -> tuple[str, dict]:
        """Return the transmon spectroscopy line and pulse parameters.

        Returns:
            line:
                The ancilla transmon drive line.
            params:
                The transmon spectroscopy pulse parameters.
        """
        param_keys = ["amplitude", "length", "pulse"]
        params = {
            k: getattr(self.parameters, f"transmon_spectroscopy_{k}")
            for k in param_keys
        }
        return "drive_transmon", params

    def swap_parameters(self) -> tuple[str, dict]:
        """Return the SWAP drive line and parameters.

        The SWAP line is a dedicated SG channel in LF mode,
        used to apply DC flux pulses that bring the transmon
        into resonance with the memory cavity.

        Returns:
            line:
                The SWAP drive line name.
            params:
                Dictionary with keys: length, amplitude, pulse.
        """
        param_keys = ["length", "amplitude", "pulse"]
        params = {k: getattr(self.parameters, f"swap_{k}") for k in param_keys}
        return "swap_drive", params

    def default_integration_kernels(self) -> list[Pulse]:
        """Return a default list of integration kernels.

        Returns:
            A list consisting of a single constant pulse with
            length equal to `readout_integration_length`.
        """
        return [
            dsl.create_pulse(
                {
                    "function": "const",
                    "length": (self.parameters.readout_integration_length),
                    "amplitude": 1.0,
                },
                name=f"integration_kernel_{self.uid}",
            ),
        ]

    def get_integration_kernels(
        self,
        kernel_pulses: list[dict]
        | Literal["default", "optimal", "continuous"]
        | None = None,
    ) -> list[Pulse]:
        """Create readout integration kernels for the bosonic qubit.

        Arguments:
            kernel_pulses:
                Custom definitions for the kernel pulses, passed as
                a list of pulse dictionaries, or one of the values
                "default", "optimal", or "continuous".

        Returns:
            A list of integration kernel pulses.
        """
        if kernel_pulses is None:
            kernel_pulses = self.parameters.readout_integration_kernels_type

        if kernel_pulses == "continuous":
            integration_kernels = None
        elif kernel_pulses == "default":
            integration_kernels = self.default_integration_kernels()
        elif kernel_pulses == "optimal":
            kernel_params = self.parameters.readout_integration_kernels
            if isinstance(kernel_params, (list, tuple)) and len(kernel_params) > 0:
                integration_kernels = [
                    dsl.create_pulse(
                        kp,
                        name=f"integration_kernel_{self.uid}",
                    )
                    for kp in kernel_params
                ]
            else:
                raise TypeError(
                    f"{self.__class__.__name__}.parameters"
                    f".readout_integration_kernels"
                    f" should be a list of pulse dictionaries."
                )
        elif isinstance(kernel_pulses, (list, tuple)) and kernel_pulses:
            integration_kernels = [
                dsl.create_pulse(
                    kp,
                    name=f"integration_kernel_{self.uid}",
                )
                for kp in kernel_pulses
            ]
        else:
            raise TypeError(
                "The readout integration kernels should be a list "
                "of pulse dictionaries or 'default', 'optimal', "
                "or 'continuous'."
            )

        return integration_kernels

    def calibration(  # noqa: C901, PLR0912, PLR0915
        self,
    ) -> Calibration:
        """Generate calibration from parameters and signal lines.

        Includes calibration entries for:
        - Memory cavity drive
        - Standard transmon drive (at n=0 frequency)
        - Photon-number-selective transmon drives (n=0..max_n)
        - Readout measure and acquire
        - SWAP drive (LF mode)

        Returns:
            calibration:
                Prefilled calibration object.
        """
        # --- Local oscillators ---

        memory_lo = None
        transmon_lo = None
        readout_lo = None

        if self.parameters.memory_lo_frequency is not None:
            memory_lo = Oscillator(
                uid=f"{self.uid}_memory_local_osc",
                frequency=self.parameters.memory_lo_frequency,
            )
        if self.parameters.transmon_lo_frequency is not None:
            transmon_lo = Oscillator(
                uid=f"{self.uid}_transmon_local_osc",
                frequency=self.parameters.transmon_lo_frequency,
            )
        if self.parameters.readout_lo_frequency is not None:
            readout_lo = Oscillator(
                uid=f"{self.uid}_readout_local_osc",
                frequency=self.parameters.readout_lo_frequency,
            )

        # --- Baseband oscillators ---

        readout_oscillator = None
        if self.parameters.readout_frequency is not None:
            readout_oscillator = Oscillator(
                uid=f"{self.uid}_readout_acquire_osc",
                frequency=self.parameters.readout_frequency,
                modulation_type=ModulationType.AUTO,
            )

        calibration_items = {}

        # --- Memory cavity drive ---

        if "drive_memory" in self.signals:
            sig_cal = SignalCalibration()
            if self.parameters.memory_drive_frequency is not None:
                sig_cal.oscillator = Oscillator(
                    uid=f"{self.uid}_memory_drive_osc",
                    frequency=self.parameters.memory_drive_frequency,
                    modulation_type=ModulationType.AUTO,
                )
            sig_cal.local_oscillator = memory_lo
            sig_cal.range = self.parameters.memory_drive_range
            calibration_items[self.signals["drive_memory"]] = sig_cal

        # --- Standard transmon drive (n=0) ---

        if "drive_transmon" in self.signals:
            sig_cal = SignalCalibration()
            freq = self.parameters.transmon_drive_frequency(0)
            if freq is not None:
                sig_cal.oscillator = Oscillator(
                    uid=f"{self.uid}_transmon_drive_osc",
                    frequency=freq,
                    modulation_type=ModulationType.AUTO,
                )
            sig_cal.local_oscillator = transmon_lo
            sig_cal.range = self.parameters.transmon_drive_range
            calibration_items[self.signals["drive_transmon"]] = sig_cal

        # --- Photon-number-selective transmon drives ---

        max_n = self.parameters.max_photon_number
        for n in range(max_n + 1):
            sig_name = self.selective_signal_name(n)
            if sig_name not in self.signals:
                continue
            sig_cal = SignalCalibration()
            freq_n = self.parameters.transmon_drive_frequency(n)
            if freq_n is not None:
                sig_cal.oscillator = Oscillator(
                    uid=f"{self.uid}_transmon_sel_n{n}_osc",
                    frequency=freq_n,
                    modulation_type=ModulationType.AUTO,
                )
            sig_cal.local_oscillator = transmon_lo
            sig_cal.range = self.parameters.transmon_drive_range
            calibration_items[self.signals[sig_name]] = sig_cal

        # --- Readout measure ---

        if "measure" in self.signals:
            sig_cal = SignalCalibration()
            if readout_oscillator is not None:
                sig_cal.oscillator = readout_oscillator
            sig_cal.local_oscillator = readout_lo
            sig_cal.range = self.parameters.readout_range_out
            calibration_items[self.signals["measure"]] = sig_cal

        # --- Readout acquire ---

        if "acquire" in self.signals:
            sig_cal = SignalCalibration()
            if readout_oscillator is not None:
                sig_cal.oscillator = readout_oscillator
            sig_cal.local_oscillator = readout_lo
            sig_cal.range = self.parameters.readout_range_in
            sig_cal.port_delay = self.parameters.readout_integration_delay
            sig_cal.threshold = (
                self.parameters.readout_integration_discrimination_thresholds
            )
            if self.parameters.readout_integration_kernels_type == "optimal":
                sig_cal.oscillator = Oscillator(
                    frequency=0,
                    modulation_type=ModulationType.SOFTWARE,
                )
            calibration_items[self.signals["acquire"]] = sig_cal

        # --- SWAP drive (LF mode, DC flux pulses) ---

        if "swap_drive" in self.signals:
            calibration_items[self.signals["swap_drive"]] = SignalCalibration(
                range=self.parameters.swap_drive_range,
                local_oscillator=Oscillator(frequency=0.0e9),
                port_mode=PortMode.LF,
            )

        return Calibration(calibration_items)
