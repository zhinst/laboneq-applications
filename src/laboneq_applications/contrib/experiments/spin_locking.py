"""This module defines the spin locking experiment.

The spin locking experiment has the following pulse sequence:

    qb --- [ prep transition ] --- [ x90_transition ] --- [ ry(delay) ] ---
    --- [ x90_transition ] --- [ measure ]

If multiple qubits are passed to the `run` workflow, the above pulses are applied
in parallel on all the qubits.
"""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING

import numpy as np
from laboneq.simple import Experiment, SweepParameter, dsl
from laboneq.workflow import WorkflowOptions, task, workflow
from laboneq.workflow.tasks import (
    compile_experiment,
    run_experiment,
)

from laboneq_applications.core import validation
from laboneq_applications.experiments.options import (
    TuneupExperimentOptions,
)

if TYPE_CHECKING:
    from laboneq.dsl.quantum.qpu import QPU
    from laboneq.dsl.session import Session
    from numpy.typing import ArrayLike

    from laboneq_applications.typing import Qubits, QubitSweepPoints


class SpinLockingExperimentOptions(TuneupExperimentOptions):
    """Base options for the resonator spectroscopy experiment.

    Additional attributes:
        refocus_pulse:
            String to define the quantum operation in-between the x90 pulses.
            Default: "y180".
    """

    pulse: dict = field(
        default_factory=lambda: {
            "function": "gaussian_square_sweep",
            "can_compress": True,
            "risefall_in_samples": 64,
        }
    )


class SpinLockingWorkflowOptions(WorkflowOptions):
    """Option for spectroscopy workflow.

    Attributes:
        create_experiment (EchoExperimentOptions):
            The options for creating the experiment.
    """

    create_experiment: SpinLockingExperimentOptions = SpinLockingExperimentOptions()


@workflow(name="spin_locking")
def experiment_workflow(
    session: Session,
    qpu: QPU,
    qubits: Qubits,
    lengths: QubitSweepPoints,
    rel_amp: float | None = None,
    options: SpinLockingWorkflowOptions | None = None,
) -> None:
    """The Hahn echo Workflow.

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
        lengths:
            The delays to sweep over for each qubit. Note that `delays` must be
            identical for qubits that use the same measure port.
        rel_amp:
            The relative amplitude specifies the spin_locking pulse amplitude
            relative to the pi-pulse amplitude. Default is None and corresponds to
            the pi pulse amplitude of the specified transition.
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
            quantum_operations=TunableTransmonOperations(),
        )
        temp_qubits = qpu.copy_qubits()
        result = experiment_workflow(
            session=session,
            qpu=qpu,
            qubits=temp_qubits,
            lengths=[[1e-6, 5e-6, 10e-6]], [1e-6, 5e-6, 10e-6]],
            options=options,
        ).run()
        ```
    """
    exp = create_experiment(
        qpu,
        qubits,
        lengths=lengths,
        rel_amp=rel_amp,
    )
    compiled_exp = compile_experiment(session, exp)
    _result = run_experiment(session, compiled_exp)


@task
@dsl.qubit_experiment
def create_experiment(
    qpu: QPU,
    qubits: Qubits,
    lengths: QubitSweepPoints,
    rel_amp: float | None = None,
    options: SpinLockingExperimentOptions | None = None,
) -> Experiment:
    """Creates a Hahn echo Experiment.

    Arguments:
        qpu:
            The qpu consisting of the original qubits and quantum operations.
        qubits:
            The qubits to run the experiments on. May be either a single
            qubit or a list of qubits.
        lengths:
            The delays to sweep over for each qubit. Note that `delays` must be
            identical for qubits that use the same measure port.
        rel_amp:
            The relative amplitude specifies the spin_locking pulse amplitude
            relative to the pi-pulse amplitude. Default is None and corresponds to
            the pi pulse amplitude of the specified transition.
        options:
            The options for building the experiment.
            See [EchoExperimentOptions] and [BaseExperimentOptions] for
            accepted options.
            Overwrites the options from [TuneupExperimentOptions] and
            [BaseExperimentOptions].

    Returns:
        experiment:
            The generated LabOne Q experiment instance to be compiled and executed.

    Raises:
        ValueError:
            If delays is not a list of numbers or array when a single qubit is passed.

    Example:
        ```python
        options = {
            "count": 10,
            "transition": "ge",
            "averaging_mode": "cyclic",
            "acquisition_type": "integration_trigger",
            "cal_traces": True,
        }
        options = TuneupExperimentOptions(**options)
        setup = DeviceSetup()
        qpu = QPU(
            setup=DeviceSetup("my_device"),
            qubits=[TunableTransmonQubit("q0"), TunableTransmonQubit("q1")],
            quantum_operations=TunableTransmonOperations(),
        )
        temp_qubits = qpu.copy_qubits()
        create_experiment(
            qpu=qpu,
            qubits=temp_qubits,
            lengths=[[1e-6, 5e-6, 10e-6], [1e-6, 5e-6, 10e-6]]
            options=options,
        )
        ```
    """
    # Define the custom options for the experiment
    opts = SpinLockingExperimentOptions() if options is None else options
    qubits, lengths = validation.validate_and_convert_qubits_sweeps(qubits, lengths)
    angle = rel_amp * np.pi if rel_amp is not None else np.pi

    qop = qpu.quantum_operations
    with dsl.acquire_loop_rt(
        count=opts.count,
        averaging_mode=opts.averaging_mode,
        acquisition_type=opts.acquisition_type,
        repetition_mode=opts.repetition_mode,
        repetition_time=opts.repetition_time,
        reset_oscillator_phase=opts.reset_oscillator_phase,
    ):
        for q, q_lengths in zip(qubits, lengths):
            with dsl.sweep(
                name=f"length_{q.uid}",
                parameter=SweepParameter(f"length_{q.uid}", q_lengths),
            ) as length:
                qop.prepare_state(q, opts.transition[0])
                qop.x90(q, opts.transition)
                qop.ry(
                    q,
                    angle=angle,
                    transition=opts.transition,
                    length=length,
                    pulse=opts.pulse,
                )
                qop.x90(q, opts.transition)
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


# had to re-define gaussian square pulse because default definition takes too much
# memory for sweeps
@dsl.pulse_library.register_pulse_functional
def gaussian_square_sweep(
    x: ArrayLike,
    risefall_in_samples: int | None = None,
    sigma: float | None = 1 / 3,
    zero_boundaries: bool | None = False,  # noqa: FBT002
    **_,
) -> None:
    """Create a square waveform with gaussian sides.

    Arguments:
        x (array):
            Samples of the pulse
        risefall_in_samples (int):
            Width of the rise/fall of the pulse in samples. Dynamically set to 10%
            (5% each) of `length` if not provided.
        sigma (float):
            Std. deviation of the Gaussian rise/fall portion of the pulse
        zero_boundaries (bool):
            Whether to zero the pulse at the boundaries

    Keyword Arguments:
        uid ([str][]): Unique identifier of the pulse
        amplitude ([float][]): Amplitude of the pulse

    Returns:
        pulse (Pulse): Gaussian square pulse.
    """
    if risefall_in_samples is not None and risefall_in_samples >= len(x):
        raise ValueError(
            "The width of the flat portion of the pulse must be smaller \
                than the total length."
        )

    if risefall_in_samples is None:
        width = 0.9
        risefall_in_samples = round(len(x) * (1 - width) / 2)

    flat_in_samples = len(x) - 2 * risefall_in_samples
    gauss_x = np.linspace(-1.0, 1.0, 2 * risefall_in_samples)
    gauss_part = np.exp(-(gauss_x**2) / (2 * sigma**2))
    gauss_sq = np.concatenate(
        (
            gauss_part[:risefall_in_samples],
            np.ones(flat_in_samples),
            gauss_part[risefall_in_samples:],
        )
    )
    if zero_boundaries:
        t_left = gauss_x[0] - (gauss_x[1] - gauss_x[0])
        delta = np.exp(-(t_left**2) / (2 * sigma**2))
        gauss_sq -= delta
        gauss_sq /= 1 - delta
    return gauss_sq
