"""Test parsing and formatting the default acquisition handles."""

from laboneq_applications.utils.handle_helpers import (
    active_reset_handle,
    calibration_trace_handle,
    parse_active_reset_handle,
    parse_calibration_trace_handle,
    parse_result_handle,
    result_handle,
)


def test_result_handle_formatting():
    """Test formatting the result_handle."""

    qubit_name = "q/0"
    handle = result_handle(qubit_name, prefix="rabi/resultat")
    assert handle == "rabi/resultat/q/0"


def test_result_handle_parsing():
    """Test parsing the result handle."""

    handle = "rabi/shots/resultat/q/0"
    qubit_name = parse_result_handle(handle, prefix="shots/resultat")
    assert qubit_name == "q/0"

    handle = "rabi/shots/result/q/0"
    qubit_name = parse_result_handle(handle, prefix="shots/resultat")
    assert qubit_name is None


def test_calibration_trace_handle_formatting():
    """Test formatting the calibration trace handle."""

    qubit_name = "q/0"
    state = "e"
    handle = calibration_trace_handle(qubit_name, state, prefix="calib/trace")
    assert handle == "calib/trace/q/0/e"


def test_calibration_trace_handle_parsing():
    """Test parsing the calibration trace handle."""

    handle = "calib/trace/q/0/e"
    qubit_name, state = parse_calibration_trace_handle(handle, prefix="calib/trace")
    assert qubit_name == "q/0"
    assert state == "e"

    handle = "calib/trace/q/0/e"
    qubit_name, state = parse_calibration_trace_handle(handle, prefix="calibration")
    assert qubit_name is None
    assert state is None

    handle = "calib/trace/q/0/"
    qubit_name, state = parse_calibration_trace_handle(handle, prefix="calib/trace")
    assert qubit_name is None
    assert state is None

    handle = "calib/trace/"
    qubit_name, state = parse_calibration_trace_handle(handle, prefix="calib/trace")
    assert qubit_name is None
    assert state is None

    handle = "calib/trace//"
    qubit_name, state = parse_calibration_trace_handle(handle, prefix="calib/trace")
    assert qubit_name is None
    assert state is None


def test_active_reset_handle_formatting():
    """Test the active_reset_handle function."""

    qubit_name = "q/0"
    tag = "0"
    handle = active_reset_handle(qubit_name, tag, prefix="active/reset")
    assert handle == "active/reset/q/0/0"


def test_active_reset_handle_parsing():
    """Test parsing the active reset handle."""

    handle = "active/reset/q/0/0"
    qubit_name, tag = parse_active_reset_handle(handle, prefix="active/reset")
    assert qubit_name == "q/0"
    assert tag == "0"

    handle = "active/reset/q/0/0"
    qubit_name, tag = parse_active_reset_handle(handle, prefix="active_reset")
    assert qubit_name is None
    assert tag is None

    handle = "active/reset/q/0/"
    qubit_name, tag = parse_active_reset_handle(handle, prefix="active/reset")
    assert qubit_name is None
    assert tag is None

    handle = "active/reset/"
    qubit_name, tag = parse_active_reset_handle(handle, prefix="active/reset")
    assert qubit_name is None
    assert tag is None

    handle = "active/reset//"
    qubit_name, tag = parse_active_reset_handle(handle, prefix="active/reset")
    assert qubit_name is None
    assert tag is None
