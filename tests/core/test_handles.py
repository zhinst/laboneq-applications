"""Test parsing and formatting the default acquisition handles."""

from laboneq_applications.core.handles import (
    active_reset_handle,
    calibration_trace_handle,
    result_handle,
)


def test_result_handle_formatting():
    """Test formatting the result_handle."""

    qubit_name = "q/0"
    handle = result_handle(qubit_name, prefix="rabi/resultat")
    assert handle == "q/0/rabi/resultat"


def test_result_handle_formatting_suffix():
    """Test formatting the result_handle with a suffix."""

    qubit_name = "q/0"
    handle = result_handle(qubit_name, prefix="rabi/resultat", suffix="best")
    assert handle == "q/0/rabi/resultat/best"


def test_calibration_trace_handle_formatting():
    """Test formatting the calibration trace handle."""

    qubit_name = "q/0"
    handle = calibration_trace_handle(qubit_name, prefix="calib/trace")
    assert handle == "q/0/calib/trace"


def test_calibration_trace_handle_formatting_state():
    """Test formatting the calibration trace handle."""

    qubit_name = "q/0"
    state = "e"
    handle = calibration_trace_handle(qubit_name, state, prefix="calib/trace")
    assert handle == "q/0/calib/trace/e"


def test_active_reset_handle_formatting():
    """Test the active_reset_handle function."""

    qubit_name = "q/0"
    handle = active_reset_handle(qubit_name, prefix="active/reset")
    assert handle == f"q/0/active/reset/{result_handle(qubit_name)}"
