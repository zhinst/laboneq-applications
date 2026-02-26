# laboneq_applications 26.4.0b2 (2026-02-27)

## Features

- The `TunableTransmon` `demo_platform(n_qubits)` and `tunable_transmon_setup(n_qubits)`
  functions were extended to support arbitrary numbers of qubits. Previously they supported
  at most six. The SHFQC and HDAWG instrument names and addresses in the setup were made
  unique by appending a count to them.

## Miscellaneous

- Added unit tests for the extraction of automation parameters in `workflow_automation`. (QRL-565)
- Add unit tests for the setting of temporary workflow parameters when running a layer. (QRL-618)
- Adds the possibility to `run_layer` with temporary workflow options provided as a dictionary and adds unit tests for the new functionality.


# laboneq_applications 26.4.0b1 (2026-02-13)

## Features

- **Internal** Added the workflow automation subclasses for the LabOne Q Automation framework, due to be officially released in v26.04.

  **Breaking change** Edited the signature of experiment workflows, such that workflow parameters are keyword only.

  Added the `evaluate_experiment` task to all experiment workflows. (QRL-570)
- Added a new `"continuous"` option for specifying kernel pulses for acquire and measurement
  integration kernels on `TunableTransmons`. The new option causes the hardware to integrate
  for the entire integration length, weighting all samples equally. This is useful for
  performing very long integrations where the pulse samples would not fit into the device
  memory.

## Documentation

- Add TWPA quantum element to reference documentation index.
- Fixed the link to the contribution guideline in the readme.

## Miscellaneous

- Corrected `.pre-commit-config.yaml` to conform to `pyproject.toml`. Previously, the pre-commit hooks and the CI gave different results. (QRL-571)
- Changed pre-commit hooks to use `include` list instead of `exclude` list, to more closely match the `pyproject.toml`. Updated the `pyproject.toml` ruff tooling to target the lowest python version: 3.10. (QRL-574)
- Removed uses of `update_quantum_elements` and replaced with `update`. (QRL-584)
- Added unit tests for create, run, and reset methods of `workflow_automation`. (QRL-605)
- Added a notebook with examples of workflow automation with inline decision logic. The examples show how to set up the decision logic in python without importing parameters from a `yaml` file. (QRL-608)
- Fix the `run_layer` method so that it updates its evaluation outputs when the nodes are run sequentially. (QRL-616)
- Added "deprecate" and "remove" types to towncrier.
- Fix misspelling of 'Bloch sphere'.
- Improve the `check:changelog` CI pipeline job, such that we try to fetch a deep clone of main, if possible. This solves the issue that occurs when a branch is so far off of main that `git fetch origin main` cannot find a common history. If we cannot fetch a deep clone, fall back to a shallow clone.
- Include notebooks in `ruff format`.
- Reformat decision logic notebook.

## Developer

- Fix CI issue which was causing latest LabOne Q release to be installed instead of the
  latest builds from `laboneq` repo's `main` branch. (QRL-596)
- Fixed renovate configuration so that it can also update image versions pulled from the
  internal docker registry mirroring ghcr.io.


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
