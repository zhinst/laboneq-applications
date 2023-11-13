import pytest
from laboneq.simple import *  # noqa: F403

from laboneq_library.automatic_tuneup.tuneup.analyzer import *  # noqa: F403
from laboneq_library.automatic_tuneup.tuneup.experiment import *  # noqa: F403
from laboneq_library.automatic_tuneup.tuneup.scan import ScanStatus

from .helpers import generate_scans, generate_scans_two_qubits


def test_scan_init(session, qubits):
    scan_single_qubit = generate_scans(1, session, qubits)[0]
    assert scan_single_qubit.status == ScanStatus.PENDING
    assert scan_single_qubit.result is None
    assert scan_single_qubit.dependencies == set()


def test_set_extra_calibration(scan_single_qubit):
    c = scan_single_qubit.experiment.get_calibration()
    qubit = scan_single_qubit.qubit_configs.get_qubits()[0]
    drive_range = c[qubit.signals["drive"]]
    acquire_range = c[qubit.signals["acquire"]]
    scan_single_qubit.set_extra_calibration(measure_range=0)

    assert c[qubit.signals["measure"]].range == 0
    # make sure drive_range and acquire range were not updated
    assert c[qubit.signals["drive"]] == drive_range
    assert c[qubit.signals["acquire"]] == acquire_range


def test_run(scan_single_qubit):
    scan_single_qubit.run(report=True, plot=False)
    assert scan_single_qubit.status == ScanStatus.FINISHED
    assert scan_single_qubit.result is not None

    scan_single_qubit.analyze()
    assert scan_single_qubit.qubit_configs[0]._analyzed_result == 1234

    scan_single_qubit.verify()
    assert scan_single_qubit.status == ScanStatus.PASSED


def test_run_complete(scan_single_qubit):
    scan_single_qubit.run_complete(report=True, plot=False)
    assert scan_single_qubit.status == ScanStatus.PASSED


@pytest.mark.parametrize("analyzer", [MockAnalyzer(), AlwaysFailedAnalyzer()])
def test_update_single_qubit(scan_single_qubit, analyzer):
    qubit_configs = scan_single_qubit.qubit_configs
    qubits = scan_single_qubit.qubits
    for conf in qubit_configs:
        conf.analyzer = analyzer
    true_qubit_value = scan_single_qubit.qubits[
        0
    ].parameters.readout_resonator_frequency

    scan_single_qubit.run(report=False, plot=False)
    scan_single_qubit.verify()
    scan_single_qubit.update()

    if isinstance(analyzer, AlwaysFailedAnalyzer):
        assert scan_single_qubit.status == ScanStatus.FAILED
        assert qubits[0].parameters.readout_resonator_frequency == true_qubit_value
    else:
        assert scan_single_qubit.status == ScanStatus.PASSED
        assert (
            qubits[0].parameters.readout_resonator_frequency
            == qubit_configs[0]._analyzed_result
            + qubits[0].parameters.readout_lo_frequency
        )


@pytest.mark.parametrize("analyzer", [MockAnalyzer(), AlwaysFailedAnalyzer()])
def test_update_user_defined_single_qubit(scan_single_qubit, analyzer):
    """Test updating a user-defined parameter of qubits"""
    qubit_configs = scan_single_qubit.qubit_configs
    qubits = scan_single_qubit.qubits
    for conf in qubit_configs:
        conf.analyzer = analyzer

    update_key = "test_key"
    qubit_configs[0].update_key = update_key

    true_qubit_value = qubits[0].parameters.user_defined[update_key]
    scan_single_qubit.run(report=False, plot=False)
    scan_single_qubit.verify()
    scan_single_qubit.update()

    assert qubit_configs[0]._update_key_in_user_defined

    if isinstance(analyzer, AlwaysFailedAnalyzer):
        # No update if the scan_single_qubit failed
        assert scan_single_qubit.status == ScanStatus.FAILED
        assert qubits[0].parameters.user_defined[update_key] == true_qubit_value
    else:
        assert scan_single_qubit.status == ScanStatus.PASSED
        assert (
            qubits[0].parameters.user_defined[update_key]
            == qubit_configs[0]._analyzed_result
        )


def test_run_two_qubits(session, qubits):
    scan = generate_scans_two_qubits(1, session, qubits)[0]
    scan.run(report=False, plot=False)

    # check exp is run and returns a (simulated) result
    assert scan.result is not None

    scan.analyze()
    assert scan.qubit_configs[0]._analyzed_result == 1234
    assert scan.qubit_configs[1]._analyzed_result == 1234

    scan.verify()
    assert scan.status == ScanStatus.PASSED


@pytest.mark.parametrize("analyzer_0", [MockAnalyzer(), AlwaysFailedAnalyzer()])
@pytest.mark.parametrize("analyzer_1", [MockAnalyzer(), AlwaysFailedAnalyzer()])
def test_update_two_qubits(session, qubits, analyzer_0, analyzer_1):
    scan = generate_scans_two_qubits(1, session, qubits)[0]
    qubit_configs = scan.qubit_configs
    qubits = scan.qubits

    qubit_configs[0].analyzer = analyzer_0
    qubit_configs[1].analyzer = analyzer_1

    scan.qubits[0].parameters.readout_resonator_frequency

    scan.run(report=False, plot=False)
    scan.verify()
    scan.update()

    # if one of the analyzer is AlwaysFailedAnalyzer, the scan should fail
    if isinstance(analyzer_0, AlwaysFailedAnalyzer) or isinstance(
        analyzer_1, AlwaysFailedAnalyzer
    ):
        assert scan.status == ScanStatus.FAILED
        # assert qubits[0].parameters.readout_resonator_frequency == true_qubit_value
    else:
        assert scan.status == ScanStatus.PASSED
        assert (
            qubits[0].parameters.readout_resonator_frequency
            == qubit_configs[0]._analyzed_result
            + qubits[0].parameters.readout_lo_frequency
        )


def test_add_deps(session, qubits):
    """
    Test adding dependencies to a scan.
    """
    scan1, scan2, scan3 = generate_scans(3, session, qubits)
    scan1.add_dependencies(scan2)
    assert scan1.dependencies == {scan2}
    scan1.add_dependencies(scan3)
    assert scan1.dependencies == {scan2, scan3}

    scan1.add_dependencies(scan2)
    assert scan1.dependencies == {scan2, scan3}

    scan1.dependencies = [scan2, scan3]
    assert scan1.dependencies == {scan2, scan3}

    scan1.dependencies = [scan2]
    assert scan1.dependencies == {scan2}

    # scan could not add itself as a dependency
    with pytest.raises(ValueError):
        scan1.add_dependencies(scan1)


def test_reset_status(scan_single_qubit):
    scan_single_qubit.run(report=False, plot=False)
    scan_single_qubit.verify()
    assert scan_single_qubit.status == ScanStatus.PASSED
    scan_single_qubit.reset_status()
    assert scan_single_qubit.status == ScanStatus.PENDING
