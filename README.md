# LabOne Q Applications Library (laboneq-applications)

The LabOne Q Applications Library is a library of experiments and analyses for various quantum computing applications, implemented using the 
Zurich Instruments [LabOne Q](https://github.com/zhinst/laboneq) software framework.

---
<!-- NOTE: Change to the public link when ready -->
**Documentation**: [Documentation](http://laboneq-applications-qccs-1d05f433a85f43634ee9a3c36e976e1ae43ad.pages.zhinst.com/manual/index.html)

---

## Contents of the Applications Library

The Applications Library currently contains the following:

### Qubit types

- Tunable Transmon qubits 

### Quantum Operations

- common operations for Tunable Transmon Qubits, such as `measure`, `acquire`, `rx`, `ry`, `rz`, etc.

### Pre-Built Experiments and Analyses 

Single-qubit calibration measurements:

- Resonator Spectroscopy
- Qubit Spectroscopy
- Amplitude Rabi
- Ramsey Interferometry 
- DRAG Quadrature-Scaling calibration
- Lifetime measurement
- Hahn echo
- Amplitude Fine
- Dispersive Shift
- IQ Blobs
- Raw Time Traces for Optimal Readout Weights


## How to Install and Use

The LabOne Q Applications Library depends on the Zurich Instruments' [LabOne Q](https://github.com/zhinst/laboneq) software framework.


## How to contribute

You can either clone or fork the `laboneq-applications` repository and use it as the working directory for your quantum experiments.

```
git clone https://github.com/zhinst/laboneq-applications.git
cd laboneq-applications
pip install -e .
```

See the [contributions guidelines](CONTRIBUTING.md) for more information. 
