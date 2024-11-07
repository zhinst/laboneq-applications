# How-to Guides - OpenQASM in LabOne Q

In the following notebooks, you'll learn how LabOne Q's [OpenQASM 3](https://openqasm.com/) integration can enable you to integrate external packages such as [Qiskit](https://qiskit.org/) seamlessly into your experiments. Depending on your specific use case and architecture, you will likely wish to modify these experiments and adapt them to your own workflow. Please get in touch at <info@zhinst.com> and we will be happy discuss your application.

## VQE with Qiskit in LabOne Q

In this [notebook](01_VQE_Qiskit.ipynb) you'll get a demonstration of first steps that you can use to perform a Variational Quantum Eigensolver (VQE) experiment with LabOne Q. Here, Qiskit is used as a convenient way to prepare the parameterized ansatz circuit, which is then converted to OpenQASM and imported and executed in LabOne Q.

## One and two Qubit RB with Qiskit in LabOne Q

In this [notebook](02_RandomizedBenchmarking_from_Qiskit.ipynb), you'll learn how you can use the [Qiskit Experiment Library](https://qiskit.org/ecosystem/experiments/apidocs/library.html) to generate one and two qubit randomized benchmarking experiments. You'll then export the generated experiment to OpenQASM, import your OpenQASM experiment into LabOne Q, compile, and simulate the output signals.
