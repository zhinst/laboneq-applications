from collections import namedtuple

import pytest
from laboneq.simple import *  # noqa: F403

from laboneq_library.automatic_tuneup.tuneup import TuneUp
from laboneq_library.automatic_tuneup.tuneup.analyzer import *  # noqa: F403
from laboneq_library.automatic_tuneup.tuneup.experiment import *  # noqa: F403
from laboneq_library.automatic_tuneup.tuneup.scan import Scan, ScanStatus


@pytest.fixture(scope="function")
def scans(qubits, device_setup):
    my_calibration = Calibration()
    my_calibration["/logical_signal_groups/q0/drive_line"] = SignalCalibration(
        oscillator=Oscillator(frequency=1.23e6),
    )
    my_calibration["/logical_signal_groups/q1/drive_line"] = SignalCalibration(
        oscillator=Oscillator(frequency=3.21e6),
    )
    device_setup.set_calibration(my_calibration)

    q0, q1 = qubits

    my_session = Session(device_setup=device_setup)
    my_session.connect(do_emulation=True)

    param = LinearSweepParameter(start=-300e6, stop=300e6, count=11)
    ResonatorSpectAnalyzerTranx()
    QubitSpecAnalyzer()

    exp_settings = {"integration_time": 10e-6, "num_averages": 2**5}

    scan_rabi = Scan(
        uid="rabi",
        session=my_session,
        qubit=q0,
        params=[param],
        update_key="readout_resonator_frequency",
        exp_fac=RSpecCwFactory,
        exp_settings=exp_settings,
    )

    scan_q_spec = Scan(
        uid="qubit_spec",
        session=my_session,
        qubit=q1,
        params=[param],
        update_key="resonance_frequency_ge",
        exp_fac=RSpecCwFactory,
        exp_settings=exp_settings,
    )

    scan_res = Scan(
        uid="resonator_spec",
        session=my_session,
        qubit=q1,
        params=[param],
        update_key="readout_resonator_frequency",
        exp_fac=RSpecCwFactory,
        exp_settings=exp_settings,
    )

    scan_res_cw = Scan(
        uid="resonator_cw",
        session=my_session,
        qubit=q1,
        params=[param],
        update_key="readout_resonator_frequency",
        exp_fac=RSpecCwFactory,
        exp_settings=exp_settings,
    )

    scan_resonator_power = Scan(
        uid="resonator_power",
        session=my_session,
        qubit=q1,
        params=[param],
        update_key="readout_resonator_frequency",
        exp_fac=RSpecCwFactory,
        exp_settings=exp_settings,
    )

    scan_isolate = Scan(
        uid="isolate",
        session=my_session,
        qubit=q1,
        params=[param],
        exp_fac=RSpecCwFactory,
        exp_settings=exp_settings,
    )
    scans_list = [
        scan_res_cw,
        scan_res,
        scan_q_spec,
        scan_rabi,
        scan_resonator_power,
        scan_isolate,
    ]
    ScanTuple = namedtuple("ScanTuple", [s.uid for s in scans_list])
    scans = ScanTuple(*scans_list)
    return scans


@pytest.fixture(scope="function")
def tuneup_single_qubit(scans):
    scans.rabi.add_dependencies(scans.qubit_spec)
    scans.qubit_spec.add_dependencies([scans.resonator_spec, scans.resonator_power])
    scans.resonator_spec.add_dependencies(scans.resonator_cw)
    scans.resonator_power.add_dependencies(scans.resonator_cw)

    tuneup = TuneUp(uid="single_qubit_tuneup", scans=[scans.rabi, scans.isolate])

    return tuneup


def test_tuneup_init(scans):
    tuneup = TuneUp(uid="single_qubit_tuneup")
    assert tuneup.uid == "single_qubit_tuneup"
    assert tuneup.scans == {}


def test_tuneup_with_scans(scans):
    scans.rabi.add_dependencies(scans.qubit_spec)
    scans.qubit_spec.add_dependencies([scans.resonator_spec, scans.resonator_power])
    scans.resonator_spec.add_dependencies(scans.resonator_cw)
    tuneup = TuneUp(uid="single_qubit_tuneup", scans=[scans.rabi])

    assert tuneup.scans == {s.uid: s for s in scans if s.uid != "isolate"}

    # Check with overlapping scans
    # qubit_spec is a dependency of rabi => list of scans should be the same
    tuneup = TuneUp(uid="tuneup", scans=[scans.rabi, scans.qubit_spec])
    assert tuneup.scans == {s.uid: s for s in scans if s.uid != "isolate"}

    # Check partial scan list
    tuneup = TuneUp(uid="single_qubit_tuneup", scans=[scans.qubit_spec])
    assert tuneup.scans == {s.uid: s for s in scans if s.uid not in ["isolate", "rabi"]}


def test_tuneup_find_required_nodes(tuneup_single_qubit):
    tuneup = tuneup_single_qubit
    scans = tuneup.scans

    assert tuneup.find_required_nodes(scans["rabi"].uid) == [
        scans["resonator_cw"].uid,
        scans["resonator_spec"].uid,
        scans["resonator_power"].uid,
        scans["qubit_spec"].uid,
    ]

    assert tuneup.find_required_nodes(scans["qubit_spec"].uid) == [
        scans["resonator_cw"].uid,
        scans["resonator_spec"].uid,
        scans["resonator_power"].uid,
    ]


def test_tuneup_run_up_to(tuneup_single_qubit):
    tuneup = tuneup_single_qubit
    scans = tuneup.scans

    tuneup.run_up_to(scans["rabi"], plot=False)
    for scan in scans.values():
        if scan.uid in ("isolate", "rabi"):
            assert scan.status == ScanStatus.PENDING
        else:
            assert scan.status == ScanStatus.PASSED

def test_tuneup_run(tuneup_single_qubit):
    tuneup = tuneup_single_qubit
    scans = tuneup.scans
    tuneup.reset_status()
    tuneup.run(scans["rabi"], plot=False)
    for scan in scans.values():
        if scan.uid == "isolate":
            assert scan.status == ScanStatus.PENDING
        else:
            assert scan.status == ScanStatus.PASSED

    tuneup.reset_status()
    scans["resonator_spec"].analyzer = AlwaysFailedAnalyzer()
    tuneup.run_up_to(scans["rabi"], plot=False)
    assert scans["rabi"].status == ScanStatus.PENDING
    assert scans["qubit_spec"].status == ScanStatus.PENDING
    assert scans["resonator_spec"].status == ScanStatus.FAILED
    assert scans["resonator_cw"].status == ScanStatus.PASSED
    assert scans["isolate"].status == ScanStatus.PENDING
