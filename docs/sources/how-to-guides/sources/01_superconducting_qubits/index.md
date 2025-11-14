# How-to Guides - Superconducting Qubits

The following guides are intended to help you become familiar with how different experiments can be written in LabOne Q. Depending on your specific use case and architecture, you will likely wish to modify these experiments and adapt them to your own use case. Please get in touch at <info@zhinst.com> and we will be happy discuss your application.


## Pulse-level Experiments

The pulse-level experiment how-to guides are intended to help you get familiar with writing and using your own pulse sequences in LabOne Q, from tune-up to advanced experiments including:

### Spectroscopy

* [Resonator Spectroscopy](02_pulse_sequences/01_tuneup/01_cw_resonator_spec_shfsg_shfqa_shfqc.ipynb)
* [Resonator Spectroscopy vs Power](02_pulse_sequences/01_tuneup//03_resonator_spec_vs_power_shfsg_shfqa_shfqc.ipynb)

### Control Pulse Tune-up

* [Rabi](02_pulse_sequences/01_tuneup/06_amplitude_rabi.ipynb)
* [Ramsey](02_pulse_sequences/01_tuneup/07_ramsey.ipynb)

### Additional Experiments

* [Active Reset](02_pulse_sequences/02_advanced_qubit_experiments/00_active_qubit_reset_shfsg_shfqa_shfqc.ipynb)
* [Randomized Benchmarking](02_pulse_sequences/02_advanced_qubit_experiments/01_randomized_benchmarking.ipynb)

Look at the navigation to see the full list of available experiments.

## Workflow-based Experiments

These experiments use the new workflow and tasks components included in LabOne Q, along with qubits and quantum operations:

* [Resonator Spectroscopy](01_workflows/01_resonator_spectroscopy.ipynb)
* [Resonator Spectroscopy DC bias](01_workflows/02_resonator_spectroscopy_dcbias.ipynb)
* [Qubit spectroscopy](01_workflows/03_qubit_spectroscopy.ipynb)
* [Amplitude Rabi](01_workflows/04_amplitude_rabi.ipynb)
* [Ramsey interferometry](01_workflows/05_ramsey.ipynb)
* [DRAG pulse characterization](01_workflows/06_drag_q_scaling.ipynb)
* [Lifetime_measurement](01_workflows/07_lifetime_measurement.ipynb)
* [Echo](01_workflows/08_echo.ipynb)
* [Amplitude Fine](01_workflows/09_amplitude_fine.ipynb)
* [Dispersive Shift](01_workflows/10_dispersive_shift.ipynb)
* [IQ blobs](01_workflows/11_iq_blobs.ipynb)
* [Raw Time Traces for Optimal Readout Weights](01_workflows/12_time_traces.ipynb)
* [Measurement QNDness](01_workflows/13_qnd_measurement.ipynb)
* [ZZ coupling strength with tunable couplers](01_workflows/14_zz_coupling.ipynb)

## Tune-Up Guides

These guides teach you how to tune-up a quantum processor with superconducting qubits.

* [Qubit Tune-Up](03_tuneup_guides/00_tuneup_qubits.ipynb)
* [Active Reset](03_tuneup_guides/01_tuneup_active_reset.ipynb)