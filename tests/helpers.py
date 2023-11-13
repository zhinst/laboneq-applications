from laboneq.dsl.result import AcquiredResult, AcquiredResults
from laboneq.simple import *

from laboneq_library.automatic_tuneup.tuneup.analyzer import *
from laboneq_library.automatic_tuneup.tuneup.experiment import *
from laboneq_library.automatic_tuneup.tuneup.params import SweepParams
from laboneq_library.automatic_tuneup.tuneup.qubit_config import (
    QubitConfig,
    QubitConfigs,
)
from laboneq_library.automatic_tuneup.tuneup.scan import Scan


def generate_scans(n, session, qubits):
    exp_settings = {"integration_time": 10e-6, "num_averages": 2**5}
    qconfigs = generate_qubit_configs(1, session, qubits)
    scans = []
    for i in range(n):
        scans.append(
            Scan(
                uid=f"scan{i}",
                session=session,
                qubit_configs=qconfigs,
                exp_fac=ResonatorCWSpec,
                exp_settings=exp_settings,
            )
        )
    return scans


def generate_scans_two_qubits(n, session, qubits):
    """Generate n scans for two qubits (corresponding to n QubitConfig) for testing purposes only.
    Default update_key is "readout_resonator_frequency".
    Default analyzer is MockAnalyzer.
    Default experiment is ParallelResSpecCW.
    """
    exp_settings = {"integration_time": 10e-6, "num_averages": 2**5}
    qconfigs = generate_qubit_configs(2, session, qubits)
    scans = []
    for i in range(n):
        scans.append(
            Scan(
                uid=f"scan{i}",
                session=session,
                qubit_configs=qconfigs,
                exp_fac=ParallelResSpecCW,
                exp_settings=exp_settings,
            )
        )
    return scans


def generate_qubit_configs(n, session, qubits):
    """Generate QubitConfigs for n qubits (corresponding to n QubitConfig) for testing purposes only. Limited to 2 for now.
    Default update_key is "readout_resonator_frequency".
    Default analyzer is MockAnalyzer.
    Default parameter is SweepParams with ONLY frequency sweep from -300e6 to 300e6 with 11 points.
    Please change the parameter afterwards if you want to test other parameters.
    """
    if n > 2:
        raise NotImplementedError("Only 1 or 2 qubits are supported for now")

    qconfigs = QubitConfigs()
    params = []
    for i in range(n):
        sweep = LinearSweepParameter(start=-300e6, stop=300e6, count=11)
        param = SweepParams(frequency=sweep)
        params.append(param)
        qconfigs.append(
            QubitConfig(
                parameter=params[i],
                qubit=qubits[i],
                update_key="readout_resonator_frequency",
                analyzer=MockAnalyzer(),
            )
        )
    return qconfigs


def sim_resonances(f0=1, amplitude=1, gamma=0.1, offset=0.1):
    x = np.linspace(-10, 10, 100)
    y = offset + amplitude * gamma**2 / (gamma * 2 + (x - f0) ** 2)

    res1 = AcquiredResult(data=y, axis_name=["frequency"], axis=[x])
    res2 = AcquiredResult(data=np.array([3, 4]), axis_name=["param1"], axis=[[0, 1]])
    results = Results(acquired_results=AcquiredResults(res_spec=res1, dummy=res2))
    return results
