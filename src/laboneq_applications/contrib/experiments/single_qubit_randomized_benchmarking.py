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

from typing import TYPE_CHECKING

import numpy as np
from laboneq.dsl.quantum import QuantumParameters
from laboneq.simple import Experiment, SweepParameter, dsl, workflow
from laboneq.workflow.tasks import (
    compile_experiment,
    run_experiment,
)
from qiskit_experiments.library import randomized_benchmarking

from laboneq_applications.contrib.analysis.single_qubit_randomized_benchmarking import (
    analysis_workflow,
)
from laboneq_applications.conversion import as_quantum_element_list
from laboneq_applications.experiments.options import (
    TuneupExperimentOptions,
    TuneUpWorkflowOptions,
)
from laboneq_applications.tasks import (
    temporary_qpu,
    temporary_quantum_elements_from_qpu,
)

if TYPE_CHECKING:
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.session import Session

    from laboneq_applications.typing import QuantumElements


@workflow.workflow(name="single_qubit_randomized_benchmarking")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: list[str] | str,
    *,
    length_cliffords: list,
    variations: int = 1,
    seed: int | None = None,
    gate_map: dict | None = None,
    temporary_parameters: dict[str, dict | QuantumParameters] | None = None,
    options: TuneUpWorkflowOptions | None = None,
) -> None:
    """The Randomized Benchmarking Workflow for single qubits.

    The workflow consists of the following steps:

    - [generate_clifford_sequences][]
    - [create_experiment][]
    - [compile_experiment][]
    - [run_experiment][]
    - [analysis_workflow][]

    !!! version-changed "Changed in version 26.10.0"
        The experiment no longer routes its Clifford sequences through OpenQASM.
        `create_sq_rb_qasm` is now `generate_clifford_sequences`, returns the
        gates themselves and applies the default `gate_map` itself, so
        `add_qasm_operations` and `get_gate_map` are gone.

    !!! version-removed "Removed in version 26.7.0"
        The `qubits` argument of type `QuantumElements` has been removed.
        Please pass `qubits` of type `list[str] | str` instead, i.e., the quantum
        element UIDs instead of the quantum element instances.

    !!! version-changed "Changed in version 26.4.0"
        All arguments apart from `session`, `qpu`, and `qubits` are now keyword
        arguments.

    !!! version-changed "Changed in version 26.1.0"
        The `temporary_parameters` positional argument was added in the
        penultimate position. Note that this is a breaking change if
        calling the experiment workflow with the `options` positional argument.

    Arguments:
        session:
            The connected session to use for running the experiment.
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubits to run the experiments on, passed by UID. May be either a single
            qubit or a list of qubits.
        length_cliffords:
            list of numbers of Clifford gates to sweep
        variations:
            Number of random seeds for RB.
        seed:
            A seed used to initialize numpy.random.default_rng when generating circuits.
            Default is None and provides a random seed.
        gate_map:
            Dictionary to define the native gate set in qiskit and the corresponding
            quantum_operations's in LabOne Q.
            Default: {"id":None, "sx":"x90", "x":"x180", "rz":"rz"}.
        temporary_parameters:
            The temporary parameters to update the qubits with.
        options:
            The options for building the workflow.
            In addition to options from
            [WorkflowOptions][laboneq.workflow.WorkflowOptions], the following
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
        # QPU from a two-qubit device setup
        qpu = QPU(
            quantum_elements=TunableTransmonQubit.from_device_setup(setup),
            quantum_operations=TunableTransmonOperations(),
        )
        result = experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=["q0", "q1"],
            length_cliffords=[1, 5, 10, 20, 50],
            variations=5,
            options=options,
        ).run()
        ```
    """
    clifford_sequences = generate_clifford_sequences(
        length_cliffords=length_cliffords,
        gate_map=gate_map,
        variations=variations,
        seed=seed,
    )

    temp_qpu = temporary_qpu(qpu, temporary_parameters)
    temp_qubits = temporary_quantum_elements_from_qpu(temp_qpu, qubits)
    exp = create_experiment(temp_qpu, temp_qubits, clifford_sequences)
    compiled_exp = compile_experiment(session, exp)
    result = run_experiment(session, compiled_exp)
    with workflow.if_(options.do_analysis):
        analysis_workflow(result, temp_qubits, length_cliffords, variations)
    workflow.return_(result)


@workflow.task
def generate_clifford_sequences(
    length_cliffords: list,
    gate_map: dict | None = None,
    variations: int = 1,
    seed: int | None = None,
    options: TuneupExperimentOptions | None = None,
) -> tuple[list[tuple], ...]:
    """Creates the RB Clifford sequences as lists of gates.

    Arguments:
        length_cliffords:
            A list of RB sequences lengths.
        gate_map:
            Dictionary to define the native gate set in qiskit and the corresponding
            quantum operations in LabOne Q.
            Default: {"sx":"x90", "x":"x180", "rz":"rz"}.
        variations:
            Number of samples to generate for each sequence length.
        seed:
            A seed used to initialize numpy.random.default_rng when generating circuits.
        options:
            The options for building the workflow.
            In addition to options from
            [WorkflowOptions][laboneq.workflow.WorkflowOptions], the following
            custom options are supported:

            - create_experiment: The options for creating the experiment.

    Returns:
        One list per Clifford sequence of `(operation, *parameters)` gates, where
            `operation` names a quantum operation.
    """
    gate_map = gate_map or {"sx": "x90", "x": "x180", "rz": "rz"}
    # create rb sequences from qiskit
    rb_experiment = randomized_benchmarking.StandardRB(
        physical_qubits=[0],
        lengths=length_cliffords,
        num_samples=variations,
        seed=seed,
    )
    rb_experiment.set_transpile_options(basis_gates=list(gate_map))

    # Barriers only order the gates, which the list already does.
    # Measure will be added later in LabOne Q DSL
    # The tuple is needed because the folder store serializes lists
    # through numpy, which cannot hold sequences of different lengths.
    return tuple(
        [
            (gate_map[g.operation.name], *(float(p) for p in g.operation.params))
            for clifford in circuit.data
            if clifford.operation.name not in ("barrier", "measure")
            for g in clifford.operation.definition.data
        ]
        for circuit in rb_experiment.circuits()
    )


@workflow.task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubits: QuantumElements,
    clifford_sequences: tuple[list[tuple], ...],
    options: TuneupExperimentOptions | None = None,
) -> Experiment:
    """Creates a single-qubit randomized benchmarking experiment.

    Arguments:
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubits to run the experiments on. May be either a single
            qubit or a list of qubits.
        clifford_sequences:
            The RB sequences that [generate_clifford_sequences][] returns.
        options:
            The options for building the workflow.
            In addition to options from
            [WorkflowOptions][laboneq.workflow.WorkflowOptions], the following
            custom options are supported:

            - create_experiment: The options for creating the experiment.

    Returns:
        experiment:
            The generated LabOne Q experiment instance to be compiled and executed.

    Raises:
        ValueError:
            If the quantum operations cannot prepare the initial state of the
            transition or one of the calibration states.

    Example:
        ```python
        options = TuneupExperimentOptions()
        options.count = 10
        # QPU from a two-qubit device setup
        qpu = QPU(
            quantum_elements=TunableTransmonQubit.from_device_setup(setup),
            quantum_operations=TunableTransmonOperations(),
        )
        clifford_sequences = generate_clifford_sequences(
            length_cliffords=[1, 5, 10, 20, 50],
            variations=5,
        )
        q0, q1 = qpu["q0"], qpu["q1"]
        create_experiment(
            qpu=qpu,
            qubits=[q0, q1],
            clifford_sequences=clifford_sequences,
            options=options,
        )
        ```
    """
    opts = TuneupExperimentOptions() if options is None else options

    # TODO: so far clifford sequences are identical for all qubits
    # finally have different clifford sequences on all qubits
    indices = np.arange(len(clifford_sequences))
    qop = qpu.quantum_operations
    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        for q in as_quantum_element_list(qubits):
            with dsl.sweep(
                name=f"rb_{q.uid}",
                parameter=SweepParameter(f"index_{q.uid}", indices),
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
                        for i, sequence in enumerate(clifford_sequences):
                            with dsl.case(i):
                                for operation, *params in sequence:
                                    qop[operation](q, *params)

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
