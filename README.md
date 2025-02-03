# LabOne Q Applications Library (laboneq-applications)

The LabOne Q Applications Library is a library of experiments and analyses for various quantum computing applications, implemented using the 
Zurich Instruments [LabOne Q](https://github.com/zhinst/laboneq) software framework.

---
**Documentation**: [Documentation](https://docs.zhinst.com/labone_q_user_manual/index.html)

---

## Contents of the Applications Library

The Applications Library currently contains the following:

### Quantum Elements

- Tunable Transmon qubits 
- Travelling-Wave Parametric Amplifiers (TWPAs)

### Quantum Operations

- common operations for Tunable Transmon Qubits, such as `measure`, `acquire`, `rx`, `ry`, `rz`, etc.
- common operations for TWPAs: `twpa_measure`, `twpa_acquire`, `set_pump_power`, `set_pump_cancellation`, etc. 

### Pre-Built Experiments and Analyses 

Single-qubit gate calibration for transmons:

- Resonator Spectroscopy
- Qubit Spectroscopy
- Amplitude Rabi
- Ramsey Interferometry 
- DRAG Quadrature-Scaling calibration
- Lifetime measurement
- Hahn echo
- Amplitude Fine

Read-out calibration for transmons:

- Dispersive Shift
- Single-Short Read-Out Characterization (IQ Blobs)
- Raw Time Traces for Optimal Readout Weights

Basic TWPA calibration using the Zurich Instruments SHF-PPC:

- TWPA-Pump Cancellation Tone Calibration
- Gain Curve Measurement
- TWPA Spectroscopy Measurement

## How to Install and Use

The LabOne Q Applications Library depends on the Zurich Instruments' [LabOne Q](https://github.com/zhinst/laboneq) software framework.

You can install it using:

```
pip install laboneq-applications
```

## How to contribute

You can either clone or fork the `laboneq-applications` repository and use it as the working directory for your quantum experiments.

```
git clone https://github.com/zhinst/laboneq-applications.git
cd laboneq-applications
pip install -e .
```

See the [contributions guidelines](CONTRIBUTING.md) for more information. 
