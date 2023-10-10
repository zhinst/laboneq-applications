# laboneq-library
Lightweight experiment execution engine for LabOne Q experiments and an extensible collection of top-quality pre-built experiments.

Currently, the `laboneq-library` is internal and serves as a learning and development platform. The 

## Goals of the library and its parts
The `laboneq-library` provides a scalable framework for quantum experiments based on the Zurich Instruments LabOne Q software framework.

To this end, the `laboneq-library` contains several elements:

### 1-qubit tuneup
### Experiment library
A library that holds basic experiment definitions. In the beginning, focused on transmons.

### Quantum operations
A library that defines quantum operations (such as pulses or gates) on a specific qubit.

### An automation library
The library of experiments and quantum operations will be used in an automation framework. With this, (tune-up) experiment workflows can be defined and run automatically.

### More elements
- Pulse functionals

## How to install and use?
The laboneq-library is used together and depends on Zurich Instruments' LabOne Q software framework.
```
pip install laboneq
```

You can clone the `laboneq-library` directory and use it as the working directory for your quantum experiments. Install the `laboneq-library` via
```
pip install -e .
```

## How to contribute?
Contribute via pull requests. There are no releases.

Pull requests must be tested in emulation mode before they are merged. Ideally, a test on quantum hardware is done, too, or at least scheduled.
