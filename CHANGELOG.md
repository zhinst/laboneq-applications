# laboneq_applications 26.1.0b4 (2026-01-15)

## Features

- Change result type in case there is only a single acquisition for a handle in the entire experiment. Before: `np.complex128`, After: `np.ndarray`.
- The `qubits`/`qubit`/`parametric_amplifier` argument of type `QuantumElements` is deprecated for all experiment workflows in v26.1.0 and will no longer be supported in v26.4.0. Please pass an argument of type `list[str] | str` instead, i.e., the quantum element UIDs instead of the quantum element instances.

  The `temporary_parameters` positional argument was added to the following `contrib` experiment workflows: `amplitude_rabi_chevron`, `signal_propagation_delay`, `single_qubit_randomized_benchmarking`, `spin_locking`, `time_rabi`, and `time_rabi_chevron`. This is a breaking change if calling these experiment workflows with the `options` positional argument.
- Changed `calibrate_cancellation` workflow to set pump cancellation attenuation and phase to 0.0 when cancellation is off.
- Support using SG channels for flux lines in `TunableTransmonQubit` and in `TunableCoupler`.

## Bug Fixes

- Fixed a bug where setting a minimum width for the Lorentzian fit was missing, causing division by zero errors.

## Documentation

- Updated the folder store documentation to clarify that only dicts of QuantumElement
  or QuantumParameters whose keys are strings or tuples of strings may be serialized
  by the folder store serializer.

## Developer

- Added @chavdard and @josepha to the list of renovate MR reviewers.
- Replaced use of mkdocs "import" configuration option with "inventories" to support new versions of mkdocs.


# laboneq_applications 26.1.0b3 (2025-12-19)

## Features

- Change result type in case there is only a single acquisition for a handle in the entire experiment. Before: `np.complex128`, After: `np.ndarray`.
- Changed `calibrate_cancellation` workflow to set pump cancellation attenuation and phase to 0.0 when cancellation is off.
- Support using SG channels for flux lines in `TunableTransmonQubit` and in `TunableCoupler`.

## Documentation

- Updated the folder store documentation to clarify that only dicts of QuantumElement
  or QuantumParameters whose keys are strings or tuples of strings may be serialized
  by the folder store serializer.

## Developer

- Added towncrier for maintaining the changelog.
