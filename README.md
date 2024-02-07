# laboneq-library
Lightweight experiment execution engine for LabOne Q experiments and an extensible collection of top-quality pre-built experiments.

**IMPORTANT: Please do not offer this package to customers without aligning with Stefania Lazar.**

- **Currently, the `laboneq-library` is internal and serves as a learning and development platform for the final product. We do not provide support for the current version.**

## Goals of the library and its parts
The `laboneq-library` provides a scalable framework for quantum experiments based on the Zurich Instruments [LabOne Q](https://github.com/zhinst/laboneq) software framework.

To this end, the `laboneq-library` contains several elements:

### Supported qubit types

- Transmon qubits

### Experiment library
A library that holds basic experiment definitions. In the beginning, focused on transmon qubits.

- Single Qubit Gate Tune-Up

    - Amplitude Rabi
    - Ramsey
    - Ramsey Parking
    - Q Scale
    - T1
    - Echo

- Signal Propagation Delay
- Resonator spectroscopy

    - Dispersive shift

- State discrimination
- Optimal integration kernels


### Quantum operations
A library that defines quantum operations (such as pulses or gates) on a specific qubit.

### More elements
- Pulse functionals

## How to install and use?

The laboneq-library is used together and depends on Zurich Instruments' LabOne Q software framework.

> :warning: The following instructions are for internal testing

### For the developer
You can either clone or fork the `laboneq-library` directory and use it as the working directory for your quantum experiments.

```
git clone https://github.com/zhinst/laboneq-library.git
cd laboneq-library
pip install -e .
```

### For the user

For the users who only need the Python library, you can install the `laboneq-library` directly from `Git`. By default the `main` branch will be installed.

```
pip install --upgrade git+https://github.com/zhinst/laboneq-library.git
or
pip install --upgrade git+https://github.com/zhinst/laboneq-library.git@<feature_branch_to_be_used>
```

## How to contribute?

See [CONTRIBUTING](CONTRIBUTING.md)
