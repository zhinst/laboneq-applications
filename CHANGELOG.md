# laboneq_applications 26.7.0b4 (2026-06-18)

## Miscellaneous

- Add missing unit tests for the DRAG quadrature-scaling calibration experiment. (QRL-781)
- Fix automation tests where the web viewer thread was not stopped upon test failure.
- Removed device options for PQSC & QHUB when using `demo_platform()`


# laboneq_applications 26.7.0b3 (2026-06-05)

## Features

- Added the following experiment and analysis workflows to `contrib`, useful for two-qubit tune-up: single-qubit phase correction for parametric CZ calibration (`para_cz_corr_sq_phases_tc`), parametric CZ flux frequency vs amplitude calibration (`para_cz_flux_freq_vs_amp_tc`), parametric CZ flux frequency vs duration calibration (`para_cz_flux_freq_vs_dur_tc`), parametric CZ pulse-duration calibration (`para_cz_flux_pulse_dur_tc`), parametric CZ frequency vs RZ phase (`para_cz_freq_vs_phase`), two-qubit randomized benchmarking (`two_qubit_rb`). To achieve this, we added the necessary parameters (`CzParameters`), operations (`TunableCouplerOperations` with `cz` method), and pulses (`modulated_flux`) for the tunable coupler.

  **Breaking change:** Changed the `name` of the ZZ coupling strength experiment workflow from `zz_coupling_strength_exp` to `zz_coupling_strength`.
- Added the bosonic qubit type, quantum operations, and demo platform, along with a selection of experiments, including: displacement calibration, memory spectroscopy, selective transmon Rx calibration, SNAP gate calibration, SWAP gate calibration, and Wigner tomography.
- Updated the VQE example to use QPU, QuantumOperations, and OpenQASMTranspiler.

  Previously the VQE example used `GateStore` and `exp_from_qasm_list` which
  are deprecated interfaces to the LabOne Q `OpenQASMTranspiler`. The `GateStore`
  is replaced with a `QPU` and a `QuantumOperations` class. `exp_from_qasm_list`
  is replaced by calling `OpenQASMTranspiler.batch_experiment` directly instead.

## Removals from the Codebase

- Removed the deprecated `update_qubits` task (use `update_qpu` instead) and the deprecated `temporary_modify` task (use `temporary_qpu` instead).

## Miscellaneous

- Fixed a bug where bosonic qubit analysis workflows had inconsistencies.
- Fixed a bug where the demo tunable transmon platform did not work for many qubits due to some parameters exceeding the allowed frequency band. Added a smoke test to verify that an experiment with 192 qubits can be compiled.

## Developer

- Fixed unintended warnings in example notebooks. The few warnings that remain are either intentional, due to emulation mode, or related to internal packages.


# laboneq_applications 26.7.0b2 (2026-05-22)

## Removals from the Codebase

- Removed support for `qubit`/`qubits`/`parametric_amplifier` arguments of type `QuantumElement`/`QuantumElements`/`TWPA` in experiment workflows. Removed support for `quantum_elements` arguments of type `QuantumElements` in the `temporary_quantum_elements_from_qpu` task. Please pass the quantum element UIDs instead of the quantum element instances.

## Developer

- Adjusted demo QPU drive LO frequency assignments to keep intermediate frequencies within the ±1 GHz range now enforced by the LabOne Q compiler.
- Fixed the failed tests after rejecting IF frequencies >= 1 GHz on SHF instruments.


# laboneq_applications 26.7.0b1 (2026-05-08)

## Developer

- CI pipelines now pick the latest builds from `main` or `release-X.Y` branches of
  laboneq that match the version within laboneq-applications commit (only
  MAJOR.MINOR.PATCH part). Note that `release-X.Y` branch builds will take precedence.
- Resolve sign for precomputed integration kernels in test fixtures (zhinst-utils update)


# laboneq_applications 26.4.0 (2026-04-30)

## Features

- Released the LabOne Q Workflow Automation framework and added a [tutorial](https://docs.zhinst.com/labone_q_user_manual/applications_library/tutorials/sources/experiment_workflow_automation.html). (QRL-551)


## Documentation

- Added a new section to the \[Experiment Workflows tutorial](https://docs.zhinst.com/labone\_q\_user\_manual/applications\_library/tutorials/sources/experiment\_workflows.html) in the Applications Library explaining how to run an experiment workflow on previously measured results saved locally as JSON files.
- Fixed incorrect docstrings across experiment and analysis modules where workflow names, parameter references, module paths, and code examples had drifted out of sync with the code.
- Updated the How-to Guides landing page in the User's manual to show the experiment workflows before the pulse sequence guides. Also added a link to the single-qubit randomized benchmarking experiment workflow documentation.

## Developer

- Configured renovate to add @skip-changelog-check statement at the beginning of the MR description.
  Developers are recommended to also add this statement in the beginning.


# laboneq_applications 26.4.0b5 (2026-04-09)

## Bug Fixes

- Fixed a bug where the functionality to determine the length of the measure
  section on `laboneq.dsl.quantum.QPU` was not compatible with all types of
  qubits. It only worked with tunable transmons.

  To avoid confusion and bugs with other types of qubits, the functionality has
  been moved to `TunableTransmonOperations.measure_section_length`.

  The method on `laboneq.dsl.quantum.QPU` will be deprecated and removed in a
  future release.

## Miscellaneous

- Changed tests to select the matplotlib backend explicitly.

  Instead of relying on the environment where the tests are run,
  a non-interactive backend is selected for the test suite.
- Simplified the `WorkflowLayer.run_executable` method.
- Fixed the overlapping of nodes in the automation web viewer. The size of the nodes is now computed dynamically based on the window size, number of layers, and number of nodes per layer.
- Moved the automation web viewer to the `laboneq` repository. It can now be imported from `laboneq.automation.web_viewer`.
- Extended the fine-amplitude test to a larger number of qubits.


# laboneq_applications 26.4.0b4 (2026-03-27)

## Features

- Added the `result_handle` argument to the `calibration_traces_rotation.calculate_qubit_population_2d` function, useful for storing multiple measurement outcomes per qubit.

## Bug Fixes

- Fixed a bug where several analysis methods did not properly support experiment results with multiple shots.

## Miscellaneous

- Updated code after replacing `__end__` with `None` in `Automation.next_layer_key`.
- Made minor improvements to the automation web viewer.

  Added the `Node results` section to the `Node info` panel.
- Renamed workflow automation parameter keys.

  **Bug fix** Fixed a bug where evaluation parameters were incorrectly assumed to be part of the experiment workflow argument list in `WorkflowLayer.run_executable`.

  Added tests for `workflow_layer`, `workflow_node`, `workflow_logic`, and `utils`.

  Added `reset` method to `WorkflowAutomation`.

  Resolved warnings in `plot_raw_complex_data_1d`.

  Softened errors to `logging.warning` so as not to interrupt an automation run.
- Renamed `_automation` to `automation`.

  Dropped `workflow_` prefix for filenames that are already in a `workflow` folder.

  Added docstrings, as mandated by ruff D103.


# laboneq_applications 26.4.0b3 (2026-03-13)

## Features

- **Breaking change** For the following non-`contrib` experiments, the `evaluation_parameters` argument has been added to replace the `evaluation_parameter`, `evaluation_parameter_thresholds`, and `evaluation_fit_r2_thresholds` arguments: `amplitude_fine`, `ampltiude_rabi`, `qubit_spectroscopy`, and `ramsey`.

  For the following non-`contrib` experiments, the `evaluation_parameters` argument, as well as a template `evaluate_experiment` task, has been added: `dispersive_shift`, `drag_q_scaling`, `echo`, `lifetime_measurement`, `resonator_spectroscopy`, and `time_traces`.

  For the following `contrib` experiments, the `evaluation_parameters` argument, as well as a template `evaluate_experiment` task, has been added: `calibrate_cancellation`, `scan_pump_parameters`, `signal_propagation_delay`, `time_rabi`, and `zz_coupling_strength_exp`.

  Consequently, all experiments with an `update_qpu` task now also have an `evaluate_experiment` task and an `evaluation_parameters` argument.

  For all experiments, all arguments apart from `session`, `qpu`, and `qubit`/`qubits`/`parametric_amplifier` are now keyword arguments.
- Refactored the workflow automation subclasses for the LabOne Q Automation framework (currently in beta), due to be officially released in v26.04.


## Bug Fixes

- Fixed a bug in the `temporary_qpu` function, where quantum elements and topology edges were not being copied correctly.


## Documentation

- Update the examples in the docstrings of the `amplitude_fine` experiment.

## Miscellaneous

- Add `serialization` notebook tutorial for LabOne Q Automation. (QRL-538)
- Visual improvements and refactoring of the automation web viewer. (QRL-647)
- Fix sequential run of layers not working with the live visualization. (QRL-648)
- Fix folder store not working with sequential run of layers in the automation framework. (QRL-650)
- Implement support for two qubit experiments in the automation framework. (QRL-652)
- Fix the folder store not working with two qubit gates when run sequentially. The folder store now creates sub-folders in the base folder `automation.timestamp-automation.name/layer.key/`. The sub-folders of the type `q1-q2/`, `q2-q3/`, ..., `qN-qM/` for two qubit gates and `q1/`, `q2/`, ..., `qN/` for single qubits. (QRL-662)
- Add live plotting to automation framework.
- Added minor improvements to automation web viewer.

## Developer

- Updated the CI job that tests against the latest `laboneq` release to test
  against the latest *beta* release if there is one.


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
