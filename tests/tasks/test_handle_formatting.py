"""Test parsing and formatting the default acquisition handles."""

from laboneq_applications.utils.handle_formatters import (
    active_reset_handle,
    calibration_trace_handle,
    result_handle,
)


def test_result_handle_formatting():
    """Test formatting the result_handle."""

    qubit_name = "q/0"
    handle = result_handle(qubit_name, prefix="rabi/resultat")
    assert handle == "rabi/resultat/q/0"


def test_calibration_trace_handle_formatting():
    """Test formatting the calibration trace handle."""

    qubit_name = "q/0"
    state = "e"
    handle = calibration_trace_handle(qubit_name, state, prefix="calib/trace")
    assert handle == "calib/trace/q/0/e"


def test_active_reset_handle_formatting():
    """Test the active_reset_handle function."""

    qubit_name = "q/0"
    tag = "0"
    handle = active_reset_handle(qubit_name, tag, prefix="active/reset")
    assert handle == "active/reset/q/0/0"
