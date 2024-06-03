# laboneq-applications

`laboneq-applications` is an experiment execution framework for LabOne Q experiments and an extensible collection of top-quality pre-built experiments for various applications.

**IMPORTANT: Please do not offer this package to customers without aligning with Stefania Lazar.**

- **Currently, the `laboneq-applications` is internal and serves as a learning and development platform for the final product. We do not provide support for the current version. If you are an application scientist and you share this with customers, you yourself are responsible for providing support until the final version is released.**

---
<!-- NOTE: Change to the public link when ready -->
**Documentation**: [Documentation](http://laboneq-applications-qccs-1d05f433a85f43634ee9a3c36e976e1ae43ad.pages.zhinst.com/manual/index.html)

---

## Goals of the Applications Library and its parts

The `laboneq-applications` provides a scalable framework for quantum experiments based on the Zurich Instruments [LabOne Q](https://github.com/zhinst/laboneq) software framework.

To this end, `laboneq-applications` contains several elements:

### Qubit types

- Transmon qubits

### Pre-built experiments

- Amplitude Rabi

### Quantum operations

A package that defines quantum operations (such as pulses or gates) on a specific qubit.

## How to install and use?

The LabOne Q Applications Library is used together and depends on Zurich Instruments' LabOne Q software framework.

<!-- NOTE: Remove when public -->
> :warning: The following instructions are for internal testing

See: [internal installation](docs/internal/install.md)

### For the developer
You can either clone or fork the `laboneq-applications` directory and use it as the working directory for your quantum experiments.

<!-- NOTE: Remove when public -->
```
git clone https://gitlab.zhinst.com/qccs/laboneq-applications.git
cd laboneq-applications
pip install -e .
```

## How to contribute?

See [CONTRIBUTING](CONTRIBUTING.md)
