from copy import deepcopy
from laboneq_applications.analysis.cal_trace_rotation import rotate_data_1d, rotate_data_2d


def extract_and_rotate_data_1d(
    results, data_handle, cal_trace_handle_root=None, cal_states="ge", do_pca=False
) -> dict:
    swpts = deepcopy(results.get_axis(data_handle)[0])
    if isinstance(swpts, list):
        swpts = swpts[0]
    data_raw = deepcopy(results.get_data(data_handle))
    if cal_trace_handle_root is None:
        cal_trace_handle_root = data_handle
    cal_trace_handles = [
        e for e in results.acquired_results if f"{cal_trace_handle_root}_cal_trace" in e
    ]
    num_cal_traces = len(cal_trace_handles)
    calibration_traces = []
    if num_cal_traces > 0:
        calibration_traces = [
            results.get_data(
                f"{cal_trace_handle_root}_cal_trace_{cal_states[0]}"
            ),
           results.get_data(
                f"{cal_trace_handle_root}_cal_trace_{cal_states[1]}"
            )
        ]
    return rotate_data_1d(raw_data=data_raw, sweep_points=swpts, calibration_traces=calibration_traces, do_pca=do_pca)


def extract_and_rotate_data_2d(
    results, data_handle, cal_trace_handle_root=None, cal_states="ge", do_pca=False
) -> dict:
    swpts_nt = deepcopy(results.get_axis(data_handle)[0])
    swpts_rt = deepcopy(results.get_axis(data_handle)[1][0])
    data_raw = deepcopy(results.get_data(data_handle))
    if cal_trace_handle_root is None:
        cal_trace_handle_root = data_handle
    cal_trace_handles = [
        e for e in results.acquired_results if f"{cal_trace_handle_root}_cal_trace" in e
    ]
    num_cal_traces = len(cal_trace_handles)
    calibration_points = []
    if num_cal_traces == 2:
        calibration_points = [
            results.get_data(
            f"{cal_trace_handle_root}_cal_trace_{cal_states[0]}"),
            results.get_data(
            f"{cal_trace_handle_root}_cal_trace_{cal_states[1]}")
        ]
    return rotate_data_2d(
        raw_data=data_raw,
        sweep_points=[swpts_nt, swpts_rt],
        calibration_traces=calibration_points,
        do_pca=do_pca
    )
