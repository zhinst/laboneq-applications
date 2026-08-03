# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines a two-qubit randomized benchmarking experiment.

In this experiment, random sequences of two-qubit Clifford gates are applied to
qubit pairs, followed by a recovery gate that should return the system to |00⟩ for
perfect gates.

The experiment has the following pulse sequence:

    qb0 --- [2q-clifford sequence] --- [recovery gate] --- [ measure ]
    qb1 --- [2q-clifford sequence] --- [recovery gate] --- [ measure ]

If multiple qubit pairs are passed, the above pulses are applied
in parallel on all pairs. The same 2-qubit Clifford sequences are applied on all pairs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from laboneq import openqasm3
from laboneq.dsl.enums import AcquisitionType, AveragingMode
from laboneq.dsl.quantum import QuantumElement, QuantumParameters
from laboneq.simple import Experiment, SweepParameter, dsl, workflow
from laboneq.workflow import option_field, task_options
from laboneq.workflow.tasks import (
    compile_experiment,
    run_experiment,
)
from qiskit import qasm3, transpile
from qiskit_experiments.library import randomized_benchmarking

from laboneq_applications.contrib.analysis.two_qubit.two_qubit_rb import (
    analysis_workflow,
)
from laboneq_applications.experiments.options import (
    TuneupExperimentOptions,
    TuneUpWorkflowOptions,
)
from laboneq_applications.tasks.parameter_updating import temporary_qpu
from laboneq_applications.typing import QuantumElements

if TYPE_CHECKING:
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.session import Session


@task_options(base_class=TuneupExperimentOptions)
class TwoQubitRBExperimentOptions:
    """Experiment options for two-qubit randomized benchmarking.

    Overrides ``averaging_mode`` and ``acquisition_type`` to use
    ``SINGLE_SHOT`` and ``DISCRIMINATION`` by default, which are required
    for two-qubit RB so that individual shot outcomes can be used to compute
    per-qubit P(q=1) statistics.
    """

    averaging_mode: AveragingMode = option_field(
        AveragingMode.SINGLE_SHOT,
        description="Averaging mode. Default: SINGLE_SHOT.",
    )
    acquisition_type: AcquisitionType = option_field(
        AcquisitionType.DISCRIMINATION,
        description="Acquisition type. Default: DISCRIMINATION.",
    )


@workflow.workflow(name="two_qubit_rb")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubit_pairs: list[list[str]],
    *,
    length_cliffords: list[int],
    variations: int = 1,
    seed: int | None = None,
    gate_map: dict | None = None,
    temporary_parameters: dict[str, dict | QuantumParameters] | None = None,
    options: TuneUpWorkflowOptions | None = None,
) -> None:
    """The Randomized Benchmarking Workflow for two-qubit pairs.

    The workflow consists of the following steps:

    - [resolve_qubit_pairs]()
    - [get_gate_map]()
    - [add_qasm_operations]()
    - [create_tq_rb_qasm]()
    - [flatten_qubit_pairs]()
    - [create_experiment]()
    - [compile_experiment]()
    - [run_experiment]()
    - [analysis_workflow]()

    Arguments:
        session:
            The connected session to use for running the experiment.
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubit_pairs:
            List of [q0, q1] qubit pairs to run the experiments on.
        length_cliffords:
            List of numbers of 2-qubit Clifford gates to sweep.
        variations:
            Number of random seeds for RB.
        seed:
            A seed used to initialize numpy.random.default_rng when generating
            circuits. Default is None and provides a random seed.
        gate_map:
            Dictionary mapping QASM gate names to quantum_operations names.
            Default: {"sx": "x90", "x": "x180", "rz": "rz", "cz": "cz"}.
        temporary_parameters:
            The temporary QPU parameters.
        options:
            The options for building the workflow.

    Returns:
        WorkflowBuilder:
            The builder for the experiment workflow.

    Example:
        ```python
        options = experiment_workflow.options()
        options.count(10)
        # QPU from a two-qubit device setup
        qpu = QPU(
            quantum_elements=TunableTransmonQubit.from_device_setup(setup),
            quantum_operations=TunableTransmonOperations(),
        )
        result = experiment_workflow(
            session=session,
            qpu=qpu,
            qubit_pairs=[["q0", "q1"]],
            length_cliffords=[1, 5, 10, 20, 50],
            variations=5,
            options=options,
        ).run()
        ```
    """
    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    qubit_pairs = resolve_qubit_pairs(temp_qpu, qubit_pairs)
    gate_map = get_gate_map(gate_map)

    quantum_operations = add_qasm_operations(temp_qpu.quantum_operations, gate_map)

    qasm_rb_sequences = create_tq_rb_qasm(
        length_cliffords=length_cliffords,
        gate_map=gate_map,
        variations=variations,
        seed=seed,
    )

    # flatten [[q0, q1], ...] → [q0, q1, coupler, ...] via a task so
    # @dsl.qubit_experiment registers all qubit and coupler signals
    all_qubits = flatten_qubit_pairs(temp_qpu, qubit_pairs)

    exp = create_experiment(
        temp_qpu,
        all_qubits,
        qubit_pairs,
        qasm_rb_sequences,
        quantum_operations=quantum_operations,
    )
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)
    with workflow.if_(options.do_analysis):
        analysis_workflow(result, qubit_pairs, length_cliffords, variations)
    workflow.return_(result)


@workflow.task
def resolve_qubit_pairs(qpu: QPU, qubit_pairs: list) -> list:
    """Resolve any string qubit UIDs in qubit_pairs to qubit objects.

    The automation framework passes qubit UIDs as strings. This task converts
    them to qubit objects so that downstream tasks can access qubit attributes.

    Arguments:
        qpu:
            The QPU used to look up qubits by UID.
        qubit_pairs:
            List of [q0, q1] pairs, where each element is either a qubit object
            or a string UID.

    Returns:
        qubit_pairs with all elements replaced by qubit objects.
    """
    return [[qpu[q] if isinstance(q, str) else q for q in pair] for pair in qubit_pairs]


@workflow.task
def get_gate_map(gate_map: dict[str, str] | None = None) -> dict[str, str]:
    """Helper task to generate the default two-qubit gate map.

    Arguments:
        gate_map:
            A dictionary specifying the names of the QASM operations as keys and
            the corresponding names in the set of quantum operations as values.
            If not provided, the default map
            {"sx": "x90", "x": "x180", "rz": "rz", "cz": "cz"} is returned.

    Returns:
        The gate map.
    """
    return (
        {"sx": "x90", "x": "x180", "rz": "rz", "cz": "cz"}
        if gate_map is None or len(gate_map) == 0
        else gate_map
    )


@workflow.task
def add_qasm_operations(
    quantum_operations: dsl.QuantumOperations,
    gate_map: dict[str, str],
) -> dsl.QuantumOperations:
    """Helper task to add QASM operation aliases to the set of quantum operations.

    Arguments:
        quantum_operations:
            The set of quantum operations to extend.
        gate_map:
            A dictionary specifying the names of the QASM operations as keys and
            the corresponding names in the set of quantum operations as values.

    Returns:
        The extended set of quantum operations.
    """
    for alias, qop_name in gate_map.items():
        quantum_operations[alias] = quantum_operations[qop_name]

    return quantum_operations


@workflow.task
def create_tq_rb_qasm(
    length_cliffords: list,
    gate_map: dict,
    variations: int = 1,
    seed: int | None = None,
    options: TwoQubitRBExperimentOptions | None = None,
) -> list:
    """Creates two-qubit RB sequences as QASM circuits.

    Arguments:
        length_cliffords:
            A list of RB sequence lengths (number of 2-qubit Cliffords).
        gate_map:
            Dictionary mapping QASM gate names to quantum operation names in LabOne Q.
        variations:
            Number of random circuit samples per sequence length.
        seed:
            A seed used to initialize numpy.random.default_rng when generating circuits.
        options:
            The options for building the workflow.

    Returns:
        A list of QASM strings, one per (length, variation) combination.
    """
    # create 2-qubit RB sequences from qiskit
    qiskit_circuits = randomized_benchmarking.StandardRB(
        physical_qubits=[0, 1],
        lengths=length_cliffords,
        num_samples=variations,
        seed=seed,
    ).circuits()

    # remove measurement from qiskit circuit
    # measurement will be added later in L1Q
    for circuit in qiskit_circuits:
        circuit.remove_final_measurements()

    # transpile to the native basis gates
    basis_gates = list(gate_map.keys())
    transpiled_circuits = transpile(
        qiskit_circuits,
        basis_gates=basis_gates,
    )

    # return QASM list of circuits
    return [qasm3.dumps(circuit) for circuit in transpiled_circuits]


@workflow.task
def flatten_qubit_pairs(qpu: QPU, qubit_pairs: list) -> list:
    """Flatten qubit pairs into a flat list including any associated couplers.

    Required so that [create_experiment]() can forward the correct flat
    element list to the ``@dsl.qubit_experiment`` decorator, which registers
    qubit *and* coupler signals in the LabOne Q experiment context.

    Arguments:
        qpu:
            The QPU, used to look up CZ-edge couplers from the topology.
        qubit_pairs:
            List of ``[q0, q1]`` pairs.

    Returns:
        Flat list of qubits and couplers:
        ``[q0, q1, coupler_q0q1, q2, q3, coupler_q2q3, ...]``.
    """
    all_elements = []
    seen_uids = set()
    for pair in qubit_pairs:
        for q in pair:
            if q.uid not in seen_uids:
                all_elements.append(q)
                seen_uids.add(q.uid)
        # include the CZ coupler so its flux signal is registered
        q0, q1 = pair[0], pair[1]
        try:
            edge = qpu.topology["cz", q0.uid, q1.uid]
            coupler = edge.quantum_element
            if coupler.uid not in seen_uids:
                all_elements.append(coupler)
                seen_uids.add(coupler.uid)
        except (KeyError, AttributeError):
            pass
    return all_elements


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubits: QuantumElements,
    qubit_pairs: list[list[QuantumElement]],
    qasm_rb_sequences: list[str],
    quantum_operations: dsl.QuantumOperations | None = None,
    options: TwoQubitRBExperimentOptions | None = None,
) -> Experiment:
    """Creates a two-qubit Randomized Benchmarking Experiment.

    Arguments:
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            Flat list of all qubits (produced by [flatten_qubit_pairs]()).
            Required by ``@dsl.qubit_experiment`` to register qubit signals in the
            LabOne Q experiment context.
        qubit_pairs:
            List of [q0, q1] qubit pairs to run the experiments on.
        qasm_rb_sequences:
            2-qubit RB sequences as QASM strings.
        quantum_operations:
            A set of quantum operations to use for the experiment.
            If None, the set from qpu.quantum_operations is used.
        options:
            The options for building the workflow.

    Returns:
        experiment:
            The generated LabOne Q experiment instance to be compiled and executed.
    """
    opts = TwoQubitRBExperimentOptions() if options is None else options

    qasm_transpiler = openqasm3.OpenQASMTranspiler(qpu)
    qop = qpu.quantum_operations if quantum_operations is None else quantum_operations

    indices = np.arange(len(qasm_rb_sequences))

    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        for pair in qubit_pairs:
            q0, q1 = pair[0], pair[1]
            pair_uid = f"{q0.uid}_{q1.uid}"

            with dsl.sweep(
                name=f"rb_{pair_uid}",
                parameter=SweepParameter(f"index_{pair_uid}", indices),
            ) as index:
                with dsl.section(name=f"prep_{pair_uid}"):
                    qop.prepare_state(q0, opts.transition[0])
                    qop.prepare_state(q1, opts.transition[0])

                with dsl.section(name=f"cliffords_{pair_uid}"):
                    with dsl.match(sweep_parameter=index):
                        for i, sequence in enumerate(qasm_rb_sequences):
                            with dsl.case(i):
                                qasm_transpiler.section(
                                    sequence, qubit_map={"q": [q0, q1]}
                                )

                with dsl.section(name=f"measure_{pair_uid}"):
                    qop.measure(q0, dsl.handles.result_handle(q0.uid))
                    qop.passive_reset(q0)
                    qop.measure(q1, dsl.handles.result_handle(q1.uid))
                    qop.passive_reset(q1)

            if opts.use_cal_traces:
                with dsl.section(name=f"cal_{pair_uid}"):
                    for q in [q0, q1]:
                        for state in opts.cal_states:
                            qop.prepare_state(q, state)
                            qop.measure(
                                q,
                                dsl.handles.calibration_trace_handle(q.uid, state),
                            )
                            qop.passive_reset(q)
