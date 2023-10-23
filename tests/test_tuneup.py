import pytest
from laboneq.simple import *  # noqa: F403

from laboneq_library.automatic_tuneup.tuneup import TuneUp
from laboneq_library.automatic_tuneup.tuneup.analyzer import *  # noqa: F403
from laboneq_library.automatic_tuneup.tuneup.experiment import *  # noqa: F403
from laboneq_library.automatic_tuneup.tuneup.scan import Scan, ScanStatus


@pytest.fixture(scope="function")
def tuneup_fixtures(qubits, device_setup):
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
        uid="q_spec",
        session=my_session,
        qubit=q1,
        params=[param],
        update_key="resonance_frequency_ge",
        exp_fac=RSpecCwFactory,
        exp_settings=exp_settings,
    )

    scan_res = Scan(
        uid="scan_res",
        session=my_session,
        qubit=q1,
        params=[param],
        update_key="readout_resonator_frequency",
        exp_fac=RSpecCwFactory,
        exp_settings=exp_settings,
    )

    scan_res_cw = Scan(
        uid="scan_res_cw",
        session=my_session,
        qubit=q1,
        params=[param],
        update_key="readout_resonator_frequency",
        exp_fac=RSpecCwFactory,
        exp_settings=exp_settings,
    )

    scan_ramsey = Scan(
        uid="scan_ramsey",
        session=my_session,
        qubit=q1,
        params=[param],
        update_key="readout_resonator_frequency",
        exp_fac=RSpecCwFactory,
        exp_settings=exp_settings,
    )

    scan_isolate = Scan(
        uid="scan_isolate",
        session=my_session,
        qubit=q1,
        params=[param],
        update_key="readout_resonator_frequency",
        exp_fac=RSpecCwFactory,
        exp_settings=exp_settings,
    )
    scans = [scan_res_cw, scan_res, scan_q_spec, scan_rabi, scan_ramsey, scan_isolate]
    return scans, my_session


def test_tuneup_init(tuneup_fixtures):
    scans, my_session = tuneup_fixtures
    scan_res_cw, scan_res, scan_q_spec, scan_rabi, scan_ramsey, scan_isolate = scans
    tuneup = TuneUp(uid="tuneup", scans=scans)

    # Add deps
    scan_rabi.add_dependencies(scan_q_spec)
    scan_q_spec.add_dependencies([scan_res, scan_ramsey])
    scan_res.add_dependencies(scan_res_cw)

    assert tuneup.uid == "tuneup"
    assert tuneup.scans == {s.uid: s for s in scans}

    # Check partial scan list
    tuneup = TuneUp(uid="tuneup", scans=[scan_rabi])
    assert tuneup.scans == {
        s.uid: s for s in [scan_rabi, scan_q_spec, scan_res, scan_ramsey, scan_res_cw]
    }

    tuneup = TuneUp(uid="tuneup", scans=[scan_rabi, scan_q_spec])
    assert tuneup.scans == {
        s.uid: s for s in [scan_rabi, scan_q_spec, scan_res, scan_ramsey, scan_res_cw]
    }


def test_tuneup_run(tuneup_fixtures):
    scans, my_session = tuneup_fixtures
    scan_res_cw, scan_res, scan_q_spec, scan_rabi, scan_ramsey, scan_isolate = scans

    scan_rabi.add_dependencies(scan_q_spec)
    scan_q_spec.add_dependencies([scan_res, scan_ramsey])
    scan_res.add_dependencies(scan_res_cw)
    scan_ramsey.add_dependencies(scan_res_cw)

    tuneup = TuneUp(uid="tuneup", scans=[scan_rabi])

    assert tuneup.find_required_nodes(scan_rabi.uid) == [
        scan_res_cw.uid,
        scan_res.uid,
        scan_ramsey.uid,
        scan_q_spec.uid,
    ]

    tuneup.run_up_to(scan_rabi, plot=False)
    assert scan_rabi.status == ScanStatus.PENDING
    assert scan_q_spec.status == ScanStatus.PASSED
    assert scan_res.status == ScanStatus.PASSED
    assert scan_ramsey.status == ScanStatus.PASSED
    assert scan_res_cw.status == ScanStatus.PASSED

    assert tuneup._run_sequence_ids == [
        scan_res_cw.uid,
        scan_res.uid,
        scan_ramsey.uid,
        scan_q_spec.uid,
    ]

    tuneup.reset_status()
    tuneup.run(scan_rabi, plot=False)
    assert scan_rabi.status == ScanStatus.PASSED
    assert scan_q_spec.status == ScanStatus.PASSED
    assert scan_res.status == ScanStatus.PASSED
    assert scan_ramsey.status == ScanStatus.PASSED
    assert scan_res_cw.status == ScanStatus.PASSED

    tuneup.reset_status()
    scan_res.analyzer = AlwaysFailedAnalyzer()
    tuneup.run_up_to(scan_rabi, plot=False)
    tuneup.display_status()
    assert scan_rabi.status == ScanStatus.PENDING
    assert scan_q_spec.status == ScanStatus.PENDING
    assert scan_res.status == ScanStatus.FAILED
    assert scan_res_cw.status == ScanStatus.PASSED
