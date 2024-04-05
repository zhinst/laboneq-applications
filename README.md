# laboneq-applications
Lightweight experiment execution engine for LabOne Q experiments and an extensible collection of top-quality pre-built experiments for various applications.

**IMPORTANT: Please do not offer this package to customers without aligning with Stefania Lazar.**

- **Currently, the `laboneq-applications` is internal and serves as a learning and development platform for the final product. We do not provide support for the current version. If you are an application scientist and you share this with customers, you yourself are responsible for providing support until the final version is released.**

## Goals of the Applications Library and its parts
The `laboneq-applications` provides a scalable framework for quantum experiments based on the Zurich Instruments [LabOne Q](https://github.com/zhinst/laboneq) software framework.

To this end, `laboneq-applications` contains several elements:

### Supported qubit types

- Transmon qubits

### Applications library
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

The LabOne Q Applications Library is used together and depends on Zurich Instruments' LabOne Q software framework.

> :warning: The following instructions are for internal testing

### For the developer
You can either clone or fork the `laboneq-applications` directory and use it as the working directory for your quantum experiments.

```
git clone https://gitlab.zhinst.com/qccs/laboneq-applications.git
cd laboneq-applications
pip install -e .
```

### For the user

For the users who only need the Python library, you can install the `laboneq-applications` directly from `Git`. By default the `main` branch will be installed.

```
pip install --upgrade git+https://gitlab.zhinst.com/qccs/laboneq-applications.git
or
pip install --upgrade git+https://gitlab.zhinst.com/qccs/laboneq-applications.git@<feature_branch_to_be_used>
```

## How to contribute?

See [CONTRIBUTING](CONTRIBUTING.md)
