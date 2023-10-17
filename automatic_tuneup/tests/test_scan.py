import pytest
from laboneq.simple import *  # noqa: F403
from tuneup.analyzer import *  # noqa: F403
from tuneup.experiment import *  # noqa: F403
from tuneup.scan import Scan, ScanStatus


def generate_scans(n, session, qubits):
    param = LinearSweepParameter(start=-300e6, stop=300e6, count=110)
    spec_analyzer = ResonatorSpectAnalyzerTranx()
    exp_settings = {"integration_time": 10e-6, "num_averages": 2**5}
    q0 = qubits[0]
    scans = []
    for i in range(n):
        scans.append(
            Scan(
                uid=f"scan{i}",
                session=session,
                qubit=q0,
                params=[param],
                update_key="readout_resonator_frequency",
                exp_fac=RSpecCwFactory,
                exp_settings=exp_settings,
                analyzer=spec_analyzer,
            )
        )
    return scans


@pytest.fixture(scope="function")
def scan(session, qubits):
    return generate_scans(1, session, qubits)[0]


def test_set_extra_calibration(scan):
    scan.set_extra_calibration(measure_range=0)
    c = scan.experiment.get_calibration()
    drive_range = c[scan.qubit.signals["drive"]]
    acquire_range = c[scan.qubit.signals["acquire"]]
    assert c[scan.qubit.signals["measure"]].range == 0

    # make sure drive_range and acquire range were not updated
    assert c[scan.qubit.signals["drive"]] == drive_range
    assert c[scan.qubit.signals["acquire"]] == acquire_range


def test_run(scan):
    scan.run(report=True, plot=False)
    assert scan.status == ScanStatus.FINISHED
    assert scan.result is not None

    scan.analyzer = DefaultAnalyzer()
    scan.analyze()
    assert scan.analyzed_result is not None

    scan.verify()
    assert scan.status == ScanStatus.PASSED


def test_run_complete(scan):
    scan.analyzer = DefaultAnalyzer()
    scan.run_complete(report=True, plot=False)
    assert scan.status == ScanStatus.PASSED
    assert scan.result is not None


def test_set_analyzer(scan):
    scan.analyzer = DefaultAnalyzer()
    assert isinstance(scan.analyzer, Analyzer)


@pytest.mark.parametrize("analyzer", [DefaultAnalyzer(), AlwaysFailedAnalyzer()])
def test_update(scan, analyzer):
    scan.analyzer = analyzer
    true_qubit_value = scan.qubit.parameters.readout_resonator_frequency
    scan.run(report=False, plot=False)
    scan.verify()
    scan.update()

    if isinstance(scan.analyzer, AlwaysFailedAnalyzer):
        assert scan.status == ScanStatus.FAILED
        assert scan.qubit.parameters.readout_resonator_frequency == true_qubit_value
    else:
        assert scan.status == ScanStatus.PASSED
        assert (
            scan.qubit.parameters.readout_resonator_frequency
            == scan.analyzed_result + scan.qubit.parameters.readout_lo_frequency
        )


@pytest.mark.parametrize("analyzer", [DefaultAnalyzer(), AlwaysFailedAnalyzer()])
def test_update_user_defined(session, qubits, analyzer):
    param = LinearSweepParameter(start=0.1, stop=1, count=110)
    exp_settings = {"integration_time": 10e-6, "num_averages": 2**5}
    q0 = qubits[0]
    scan = Scan(
        uid=f"scan0",
        session=session,
        qubit=q0,
        params=[param],
        update_key="pi_pulse_amplitude",
        exp_fac=AmplitudeRabi,
        exp_settings=exp_settings,
        analyzer=analyzer,
    )

    true_qubit_value = scan.qubit.parameters.user_defined["pi_pulse_amplitude"]
    scan.run(report=False, plot=False)
    scan.verify()
    scan.update()

    assert scan._update_key_in_user_defined == True

    if isinstance(scan.analyzer, AlwaysFailedAnalyzer):
        assert scan.status == ScanStatus.FAILED
        assert (
            scan.qubit.parameters.user_defined["pi_pulse_amplitude"] == true_qubit_value
        )
    else:
        assert scan.status == ScanStatus.PASSED
        assert (
            scan.qubit.parameters.user_defined["pi_pulse_amplitude"]
            == scan.analyzed_result
        )


def test_force_update(scan):
    scan.analyzer = AlwaysFailedAnalyzer()
    scan.update(force_value=0)
    assert scan.qubit.parameters.readout_resonator_frequency == 0


def test_add_deps(session, qubits):
    scan1, scan2, scan3 = generate_scans(3, session, qubits)
    scan1.add_dependencies(scan2)
    assert scan1.dependencies == {scan2}
    scan1.add_dependencies(scan3)
    assert scan1.dependencies == {scan2, scan3}

    scan1.dependencies = [scan2, scan3]
    assert scan1.dependencies == {scan2, scan3}

    with pytest.raises(ValueError):
        scan1.add_dependencies(scan1)
