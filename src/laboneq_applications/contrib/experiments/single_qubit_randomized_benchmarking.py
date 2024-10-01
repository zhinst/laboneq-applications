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

from laboneq.openqasm3.gate_store import GateStore
from laboneq.openqasm3.openqasm3_importer import OpenQasm3Importer
from laboneq.simple import Experiment, SweepParameter
from qiskit import qasm3, transpile
from qiskit_experiments.library import randomized_benchmarking

from laboneq_applications import dsl
from laboneq_applications.contrib.analysis.single_qubit_randomized_benchmarking import (
    analysis_workflow,
)
from laboneq_applications.core import handles
from laboneq_applications.core.build_experiment import qubit_experiment
from laboneq_applications.core.validation import validate_and_convert_qubits_sweeps
from laboneq_applications.experiments.options import (
    TuneupExperimentOptions,
    TuneUpWorkflowOptions,
)
from laboneq_applications.tasks import compile_experiment, run_experiment
from laboneq_applications.workflow import (
    if_,
    task,
    workflow,
)

if TYPE_CHECKING:
    from laboneq.dsl.session import Session

    from laboneq_applications.qpu_types import QPU
    from laboneq_applications.typing import Qubits


@workflow(name="single_qubit_randomized_benchmarking")
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
            qop's in LabOne Q.
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
        qpu = QPU(
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            qop=TunableTransmonOperations(),
        )
        temp_qubits = qpu.copy_qubits()
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
    if gate_map.isEmpty:
        gate_map = {"id": None, "sx": "x90", "x": "x180", "rz": "rz"}

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
        gate_map=gate_map,
    )
    compiled_exp = compile_experiment(session, exp)
    _result = run_experiment(session, compiled_exp)
    with if_(options.do_analysis):
        analysis_workflow(_result, qubits, length_cliffords, variations)


@task
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


@task
@qubit_experiment
def create_experiment(
    qpu: QPU,
    qubits: Qubits,
    qasm_rb_sequences: list,
    gate_map: dict,
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
        gate_map:
            Dictionary to define the native gate set in QASM and the corresponding
            qop's in LabOne Q.
            Default: {"id":None, "sx":"x90", "x":"x180", "rz":"rz"}.
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
        setup = DeviceSetup()
        qpu = QPU(
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            qop=TunableTransmonOperations(),
        )
        temp_qubits = qpu.copy_qubits()
        create_experiment(
            qpu=qpu,
            qubits=temp_qubits,
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
        indices = [range(len(qasm_rb_sequences)) for q in qubits]
    else:
        indices = range(len(qasm_rb_sequences))
    qubits, indices = validate_and_convert_qubits_sweeps(qubits, indices)

    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        for q, q_indices in zip(qubits, indices):
            # single qubit in qasm is labelled as 'q[0]'
            qubit_map = {"q[0]": q}
            gate_store = create_gate_store(qpu, qubit_map, gate_map)

            with dsl.sweep(
                name=f"rb_{q.uid}",
                parameter=SweepParameter(f"index_{q.uid}", q_indices),
            ) as index:
                with dsl.section(
                    name=f"prep_{q.uid}",
                ):
                    qpu.qop.prepare_state(q, opts.transition[0])

                with dsl.section(
                    name=f"cliffords_{q.uid}",
                ):
                    with dsl.match(
                        sweep_parameter=index,
                    ):
                        for i, sequence in enumerate(qasm_rb_sequences):
                            with dsl.case(i) as c:
                                importer = OpenQasm3Importer(
                                    qubits=qubit_map,
                                    gate_store=gate_store,
                                )
                                c.add(importer(text=sequence))

                with dsl.section(
                    name=f"measure_{q.uid}",
                ):
                    qpu.qop.measure(q, handles.result_handle(q.uid))
                    qpu.qop.passive_reset(q)

            if opts.use_cal_traces:
                with dsl.section(
                    name=f"cal_{q.uid}",
                ):
                    for state in opts.cal_states:
                        qpu.qop.prepare_state(q, state)
                        qpu.qop.measure(
                            q,
                            handles.calibration_trace_handle(q.uid, state),
                        )
                        qpu.qop.passive_reset(q)


def create_gate_store(
    qpu: QPU,
    qubit_map: dict,
    gate_map: dict,
) -> GateStore:
    """Creates a GateStore to convert QASM gates to L1Q qops.

    Arguments:
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubit_map:
            A dictionary that translates a qasm qubit to L1Q qubit.
        gate_map:
            A dictionary that translates the native gates from QASM to L1Q, e.g.
            {"id":None, "sx":"x90", "x":"x180", "rz":"rz"}
    """
    gate_store = GateStore()
    # gates
    for oq3_qubit, l1q_qubit in qubit_map.items():
        for qasm_gate, l1q_gate in gate_map.items():
            gate_store.register_gate_section(
                qasm_gate,
                (oq3_qubit,),
                lambda *args, qubit=l1q_qubit, gate=l1q_gate: qpu.qop[gate](
                    qubit, *args
                ),
            )
    return gate_store
