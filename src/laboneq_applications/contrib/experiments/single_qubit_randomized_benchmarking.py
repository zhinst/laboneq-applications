# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines a randomized benchmarking experiment.

In this experiment, ...

The experiment has the following pulse sequence:

    qb --- [clifford sequence] --- [recovery gate] --- [ measure ]

If multiple qubits are passed to the `run` workflow, the above pulses are applied
in parallel on all the qubits.

Note that in the current implementation, the same Clifford sequences are applied on all
qubits.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from laboneq import openqasm3
from laboneq.simple import Experiment, SweepParameter, dsl, workflow
from laboneq.workflow.tasks import (
    compile_experiment,
    run_experiment,
)
from qiskit import qasm3, transpile
from qiskit_experiments.library import randomized_benchmarking

from laboneq_applications.contrib.analysis.single_qubit_randomized_benchmarking import (
    analysis_workflow,
)
from laboneq_applications.core.validation import validate_and_convert_qubits_sweeps
from laboneq_applications.experiments.options import (
    TuneupExperimentOptions,
    TuneUpWorkflowOptions,
)

if TYPE_CHECKING:
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.session import Session

    from laboneq_applications.typing import Qubits


@workflow.workflow(name="single_qubit_randomized_benchmarking")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: Qubits,
    length_cliffords: list,
    variations: int = 1,
    seed: int | None = None,
    gate_map: dict | None = None,
    options: TuneUpWorkflowOptions | None = None,
) -> None:
    """The Randmized Benchmarking Workflow for single qubits.

    The workflow consists of the following steps:

    - [create_experiment]()
    - [compile_experiment]()
    - [run_experiment]()

    Arguments:
        session:
            The connected session to use for running the experiment.
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubits to run the experiments on. May be either a single
            qubit or a list of qubits.
        length_cliffords:
            list of numbers of Clifford gates to sweep
        variations:
            Number of random seeds for RB.
        seed:
            A seed used to initialize numpy.random.default_rng when generating circuits.
            Default is None and provides a random seed.
        gate_map:
            Dictionary to define the native gate set in QASM and the corresponding
            quantum_operations's in LabOne Q.
            Default: {"id":None, "sx":"x90", "x":"x180", "rz":"rz"}.
        options:
            The options for building the workflow.
            In addition to options from [WorkflowOptions], the following
            custom options are supported:
                - create_experiment: The options for creating the experiment.

    Returns:
        result:
            The result of the workflow.

    Example:
        ```python
        options = experiment_workflow.options()
        options.count(10)
        options.transition("ge")
        result = experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=temp_qubits,
            length_cliffords=[1,5,10,20,50],
            variations=5,
            options=options,
        ).run()
        ```
    """
    gate_map = get_gate_map(gate_map)

    quantum_operations = add_qasm_operations(qpu.quantum_operations, gate_map)

    qasm_rb_sequences = create_sq_rb_qasm(
        length_cliffords=length_cliffords,
        gate_map=gate_map,
        variations=variations,
        seed=seed,
    )

    exp = create_experiment(
        qpu,
        qubits,
        qasm_rb_sequences,
        quantum_operations=quantum_operations,
    )
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)
    with workflow.if_(options.do_analysis):
        analysis_workflow(result, qubits, length_cliffords, variations)
    workflow.return_(result)


@workflow.task
def get_gate_map(gate_map: dict[str, str] | None = None) -> dict[str, str]:
    """Helper task to generate the default gate map.

    Args:
        gate_map: a dictionary specifying the names of the qasm operations as keys
            and the corresponding names in the set of quantum operations as values.
            If this is not provided, the default map
            {"sx": "x90", "x": "x180", "rz": "rz"} is returned.

    Returns:
        the gate map
    """
    return (
        {"sx": "x90", "x": "x180", "rz": "rz"}
        if gate_map is None or len(gate_map) == 0
        else gate_map
    )


@workflow.task
def add_qasm_operations(
    quantum_operations: dsl.QuantumOperations,
    gate_map: dict[str, str],
) -> dsl.QuantumOperations:
    """Helper task to add qasm operations to the set of quantum operations.

    The qasm operations are added as aliases of existing operations in the set.

    Args:
        quantum_operations: the set of quantum operations to add to.
        gate_map: a dictionary specifying the names of the qasm operations as keys
            and the corresponding names in the set of quantum operations as values.

    Returns:
        the extended set of quantum operations
    """
    for alias, qop_name in gate_map.items():
        quantum_operations[alias] = quantum_operations[qop_name]

    return quantum_operations


@workflow.task
def create_sq_rb_qasm(
    length_cliffords: list,
    gate_map: dict,
    variations: int = 1,
    seed: int | None = None,
    options: TuneupExperimentOptions | None = None,
) -> list:
    """Creates RB sequences as QASM experiments.

    Arguments:
        length_cliffords:
            A list of RB sequences lengths.
        gate_map:
            Dictionary to define the native gate set in QASM and the corresponding
            quantum operations in LabOne Q.
        variations:
            Number of samples to generate for each sequence length.
        seed:
            A seed used to initialize numpy.random.default_rng when generating circuits.
        options:
            The options for building the workflow.
            In addition to options from [WorkflowOptions], the following
            custom options are supported:
                - create_experiment: The options for creating the experiment.

    Returns:
        experiment:
            The generated LabOne Q experiment instance to be compiled and executed.
    """
    # create rb sequences from qiskit
    qiskit_circuits = randomized_benchmarking.StandardRB(
        physical_qubits=[0],
        lengths=length_cliffords,
        num_samples=variations,
        seed=seed,
    ).circuits()

    # remove measurement from qiskit circuit
    # measurement will be added later again in L1Q
    for circuit in qiskit_circuits:
        circuit.remove_final_measurements()

    # transpile to basis gates
    basis_gates = list(gate_map.keys())
    transpiled_circuits = transpile(
        qiskit_circuits,
        basis_gates=basis_gates,
    )

    # return QASM list of circuits
    return [qasm3.dumps(circuit) for circuit in transpiled_circuits]


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubits: Qubits,
    qasm_rb_sequences: list,
    quantum_operations: dsl.QuantumOperations | None = None,
    options: TuneupExperimentOptions | None = None,
) -> Experiment:
    """Creates an Amplitude Rabi Experiment.

    Arguments:
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubits to run the experiments on. May be either a single
            qubit or a list of qubits.
        qasm_rb_sequences:
            RB sequences as QASM experiments.
        quantum_operations:
            A set of quantum operations to use for the experiment.
            If None, the set from qpu.quantum_operations is used.
        options:
            The options for building the workflow.
            In addition to options from [WorkflowOptions], the following
            custom options are supported:
                - create_experiment: The options for creating the experiment.

    Returns:
        experiment:
            The generated LabOne Q experiment instance to be compiled and executed.

    Raises:
        ValueError:

    Example:
        ```python
        options = TuneupExperimentOptions()
        options.count = 10
        create_experiment(
            qpu=qpu,
            qubits=qubits,
            length_cliffords=[1,5,10,20,50],
            variations=5,
            options=options,
        )
        ```
    """
    opts = TuneupExperimentOptions() if options is None else options

    # TODO: so far clifford sequences are identical for all qubits
    # finally have different clifford sequences on all qubits
    if isinstance(qubits, Sequence):
        indices = [range(len(qasm_rb_sequences)) for _ in qubits]
    else:
        indices = range(len(qasm_rb_sequences))
    qubits, indices = validate_and_convert_qubits_sweeps(qubits, indices)
    qasm_transpiler = openqasm3.OpenQASMTranspiler(qpu)
    qop = qpu.quantum_operations if quantum_operations is None else quantum_operations
    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        for q, q_indices in zip(qubits, indices):
            with dsl.sweep(
                name=f"rb_{q.uid}",
                parameter=SweepParameter(f"index_{q.uid}", q_indices),
            ) as index:
                with dsl.section(
                    name=f"prep_{q.uid}",
                ):
                    qop.prepare_state(q, opts.transition[0])

                with dsl.section(
                    name=f"cliffords_{q.uid}",
                ):
                    with dsl.match(
                        sweep_parameter=index,
                    ):
                        for i, sequence in enumerate(qasm_rb_sequences):
                            with dsl.case(i) as c:
                                qasm_section = qasm_transpiler.section(
                                    sequence, qubit_map={"q": [q]}
                                )
                                c.add(qasm_section)

                with dsl.section(
                    name=f"measure_{q.uid}",
                ):
                    qop.measure(q, dsl.handles.result_handle(q.uid))
                    qop.passive_reset(q)

            if opts.use_cal_traces:
                with dsl.section(
                    name=f"cal_{q.uid}",
                ):
                    for state in opts.cal_states:
                        qop.prepare_state(q, state)
                        qop.measure(
                            q,
                            dsl.handles.calibration_trace_handle(q.uid, state),
                        )
                        qop.passive_reset(q)
