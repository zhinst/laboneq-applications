# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines a two-qubit interleaved randomized benchmarking experiment.

The Qiskit randomized benchmarking library is used to create the IRB experiment.

The experiment has the following pulse sequence:

q0 --- [q0 clifford_sequence_0] --- [recovery gate] --- [ measure ]
q1 --- [q1 clifford_sequence_0] --- [recovery gate] --- [ measure ]

q0 --- [q0 clifford_sequence_0 w/ interleaved gate] --- [recovery gate] --- [ measure ]
q1 --- [q1 clifford_sequence_0 w/ interleaved gate] --- [recovery gate] --- [ measure ]
...
...
q0 --- [q0 clifford_sequence_N] --- [recovery gate] --- [ measure ]
q1 --- [q1 clifford_sequence_N] --- [recovery gate] --- [ measure ]

q0 --- [q0 clifford_sequence_N w/ interleaved gate] --- [recovery gate] --- [ measure ]
q1 --- [q1 clifford_sequence_N w/ interleaved gate] --- [recovery gate] --- [ measure ]

where N is the number of chosen variations.

"""

from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING, Literal

from laboneq import openqasm3
from laboneq.simple import (
    AveragingMode,
    Experiment,
    dsl,
    workflow,
)
from laboneq.workflow import if_
from laboneq.workflow.tasks import (
    compile_experiment,
    run_experiment,
)
from qiskit import qasm3, transpile
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit_experiments.library import randomized_benchmarking

from laboneq_applications.contrib.analysis.two_qubit.two_qubit_irb import (
    analysis_workflow,
)
from laboneq_applications.core.validation import (
    validate_flux_driven_edges,
)
from laboneq_applications.qpu_types.tunable_transmon import (
    TunableTransmonQubitParameters,
)
from laboneq_applications.tasks import extract_edges_from_qubit_pairs
from laboneq_applications.tasks.parameter_updating import (
    temporary_qpu,
)

if TYPE_CHECKING:
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.session import Session
    from numpy.typing import ArrayLike

    from laboneq_applications.typing import QuantumElements


@workflow.workflow_options
class TwoQubitCouplerIRBWorkflowOptions:
    """Options for the two-qubit coupler IRB experiment workflow."""

    do_analysis: bool = workflow.option_field(default=True)


@workflow.task_options
class InterleavedGateOptions:
    """Options for the tasks that need to know the interleaved gate.

    Shared by the create_irb_sequences and register_qasm_operations tasks.

    Attributes:
       interleaved_gate:
           The gate to interleave within the RB sequence. Must be registered
           as a quantum operation. The name of the quantum operation must
           follow the qiskit standard gate name mapping.
           Currently can be one of:

           - `"iswap"`
           - `"cz"`
           - `"cx"`

           Default: `"cz"`
    """

    interleaved_gate: Literal["iswap", "cz", "cx"] = workflow.option_field(
        "cz",
        description="The gate to benchmark with interleaved randomized benchmarking.",
    )


@workflow.task_options
class MultiQASMProgramOptions(openqasm3.MultiProgramOptions):
    """LabOne Q Experiment options for multiple OpenQASM programs.

    Adds the interleaved gate to the options of
    [MultiProgramOptions][laboneq.openqasm3.MultiProgramOptions], which
    documents all the other fields. The analysis only handles data taken with
    [AveragingMode.SINGLE_SHOT][] and [AcquisitionType.DISCRIMINATION][], so
    set `acquisition_type` accordingly.

    Attributes:
        averaging_mode:
            The mode of how to average the acquired data.

            Default: `AveragingMode.SINGLE_SHOT`.
        interleaved_gate:
            The gate to interleave within the RB sequence. Must be registered
            as a quantum operation under its qiskit standard gate name.

            Default: `"cz"`.
    """

    averaging_mode: AveragingMode = workflow.option_field(
        AveragingMode.SINGLE_SHOT,
        description="The mode of how to average the acquired data.",
        converter=AveragingMode,
    )
    interleaved_gate: Literal["iswap", "cz", "cx"] = workflow.option_field(
        "cz",
        description="The gate to benchmark with interleaved randomized benchmarking.",
    )


@workflow.workflow(name="two_qubit_coupler_irb")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubit_pairs: list[list[str]],
    length_cliffords: list[int] | ArrayLike,
    variations: int,
    gate_map: dict | None = None,
    temporary_parameters: dict[str, dict | TunableTransmonQubitParameters]
    | None = None,
    options: TwoQubitCouplerIRBWorkflowOptions | None = None,
) -> None:
    """The IRB Workflow for two qubits coupled by a tunable coupler.

    The workflow consists of the following steps:

    - [create_experiment][]
    - [compile_experiment][]
    - [run_experiment][]
    - [analysis_workflow][]

    Arguments:
        session:
            The connected session to use for running the experiment.
        qpu:
            The qpu consisting of the original qubits, tunable coupler and
            quantum operations.
        qubit_pairs:
            The pairs of qubits to run the experiments on. Each pair must be a
            list of two qubit UIDs.
        length_cliffords:
            list of the numbers of Clifford gates to sweep
        variations:
            Number of random seeds for the RB Clifford generation.
        gate_map:
            Dictionary to map the name of the basis gate set in QASM to the name
            of the corresponding quantum_operation in LabOne Q. Only needed if
            the quantum operation name of the interleaved gate differs from the
            standard QASM name for that gate. Or, if the basis gates or names
            differ from
            {"sx": "x90","x" : "x180","reset":"active_reset", "barrier": "barrier"}
            Default: None.
        temporary_parameters:
            The temporary parameters with which to update the qubit-pair
            parameters for the duration of the experiment.
        options:
            The options for building the workflow, passed as an instance of
            [TwoQubitCouplerIRBWorkflowOptions][].
            In addition to options from
            [WorkflowOptions][laboneq.workflow.WorkflowOptions], the following
            custom options are supported:

            - create_irb_sequences: The options for creating the interleaved
              RB sequences, as an instance of [InterleavedGateOptions][].
            - register_qasm_operations: The options for registering the
              OpenQASM operation aliases, as an instance of
              [InterleavedGateOptions][].
            - create_qasm_batch_experiment: The options for creating the
              experiment, as an instance of [MultiQASMProgramOptions][].


    Returns:
        result:
            The result of the workflow.

    Example:
        ```python
        options = experiment_workflow.options()
        options.count(10)
        options.interleaved_gate("iswap")
        result = experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=temp_qubits,
            coupler=c03
            length_cliffords=[1,5,10,20,50],
            variations=5,
            options=options,
        ).run()
        ```
    """
    # preparation
    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    qasm_program = create_irb_sequences(length_cliffords, variations)
    qubit_map = create_qasm_qubit_map(qubit_pairs)
    register_qasm_operations(temp_qpu, gate_map)
    # experiment
    experiment = create_qasm_batch_experiment(
        temp_qpu, qubit_pairs, qasm_program, qubit_map
    )
    compiled_experiment = compile_experiment(session, experiment)
    results = run_experiment(session, compiled_experiment)
    # analysis
    with if_(options.do_analysis):
        analysis_workflow(results, qpu, qubit_pairs, length_cliffords, variations)
    workflow.return_(results)


@workflow.task
def create_irb_sequences(
    length_cliffords: list[int] | ArrayLike,
    variations: int,
    options: InterleavedGateOptions | None = None,
) -> list[str]:
    """Create the interleaved RB sequences as OpenQASM programs."""
    opts = InterleavedGateOptions() if options is None else options
    interleaved_gate = get_standard_gate_name_mapping()[opts.interleaved_gate]

    rb_qiskit_circuits = randomized_benchmarking.InterleavedRB(
        interleaved_element=interleaved_gate,
        physical_qubits=[0, 1],
        lengths=length_cliffords,
        num_samples=variations,
    ).circuits()

    for circuit in rb_qiskit_circuits:
        circuit.remove_final_measurements()

    basis_gates = ["id", "sx", "x", "rz", opts.interleaved_gate]

    rb_transpiled_circuits = transpile(
        rb_qiskit_circuits,
        basis_gates=basis_gates,
        optimization_level=1,
    )

    # declaring the basis gates instead of including the standard gate library keeps
    # qiskit from writing out the definition of a gate the library does not declare,
    # such as iswap, which LabOne Q cannot import
    return [
        qasm3.dumps(circuit, includes=(), basis_gates=["U", *basis_gates])
        for circuit in rb_transpiled_circuits
    ]


@workflow.task
def create_qasm_qubit_map(
    qubit_pairs: list[list[str, str]],
) -> dict[str, QuantumElements]:
    """Create a map from OpenQASM qubit names to the flattened qubit pairs."""
    # flatted qubit_pairs
    qubits = list(chain.from_iterable(qubit_pairs))
    return {"q": qubits}


@workflow.task
def register_qasm_operations(
    qpu: QPU,
    gate_map: dict[str, str] | None,
    options: InterleavedGateOptions | None = None,
) -> dsl.QuantumOperations:
    """Task to add qasm operations to the set of quantum operations.

    The qasm operations are added as aliases of existing operations in the set.

    Args:
        qpu: The qpu consisting of the original qubits, tunable coupler and
             quantum operations. The interleaved gate must already be
             registered as a quantum_operation of the qpu.
        gate_map: a dictionary specifying the names of the qasm operations as keys
                  the corresponding names in the set of quantum operations as values.
        options: Options for this task, passed as an instance of
                 [InterleavedGateOptions][]. See the docstring of this class
                 for more details.

    Returns:
        the extended set of quantum operations
    """
    opts = InterleavedGateOptions() if options is None else options
    quantum_operations = qpu.quantum_operations

    std_gate_map = {
        "sx": "x90",
        "x": "x180",
        "reset": "active_reset",
        "barrier": "barrier",
    }
    interleaved_gate = get_standard_gate_name_mapping()[opts.interleaved_gate].name
    std_gate_map.update({interleaved_gate: interleaved_gate})
    if gate_map is not None:
        std_gate_map.update(gate_map)

    gate_map = std_gate_map

    for alias, qop_name in gate_map.items():
        quantum_operations[alias] = quantum_operations[qop_name]


@workflow.task
def create_qasm_batch_experiment(
    qpu: QPU,
    qubit_pairs: list[list[str, str]],
    programs: list[str],
    qubit_map: dict[str, QuantumElements],
    options: MultiQASMProgramOptions | None = None,
) -> Experiment:
    """A task to create a QASM batch experiment.

    Arguments:
        qpu:
            The quantum processing unit to create the experiment for.
        qubit_pairs:
            The pairs of qubits to run the experiments on. Each pair must be a
            list of two qubit UIDs.
        programs:
            List of OpenQASM programs.
        qubit_map:
            A map from OpenQASM qubit names to LabOne Q DSL Qubit objects.
            The values can either be a single qubit or a list of qubits in case of a
            qubit register.
        options:
            Options for this task, passed as in instance of [MultiQASMProgramOptions][].
            See the docstring of this class for more details.

    Returns:
        A LabOne Q Experiment.

    Raises:
        ValueError: Supplied qubit(s) does not exists in the QPU.
        OpenQasmException: The program cannot be transpiled.
    """
    transpiler = openqasm3.OpenQASMTranspiler(qpu)
    opts = MultiQASMProgramOptions() if options is None else options

    experiment = transpiler.batch_experiment(
        programs=programs,
        qubit_map=qubit_map,
        options=opts,
    )

    # add quantum elements
    edges = extract_edges_from_qubit_pairs(
        qpu,
        qubit_pairs,
        tag=opts.interleaved_gate,
    )
    validate_flux_driven_edges(edges)
    for e in edges:
        dsl.add_quantum_elements(e.quantum_element, experiment=experiment)

    return experiment
