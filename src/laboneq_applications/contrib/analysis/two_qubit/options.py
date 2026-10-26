# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Options for the contrib two-qubit analysis workflows."""

from __future__ import annotations

from typing import Literal

from laboneq import workflow

from laboneq_applications.analysis.options import (
    FitDataOptions,
)


@workflow.task_options(base_class=FitDataOptions)
class FitData2QOptions:
    """Options for the `fit_data` tasks of the two-qubit analyses.

    See [FitDataOptions][] for additional accepted options.

    Attributes:
        transition:
            Transition to perform the experiment on. May be any transition
            supported by the quantum operations.
            Default: `"ge"`.
        do_pca:
            Whether to perform principal component analysis on the raw data
            independent of whether there were calibration traces in the
            experiment.
            Default: `False`.
        use_cal_traces:
            Whether to include calibration traces in the experiment.
            Default: `True`.
    """

    transition: Literal["ge", "ef"] = workflow.option_field(
        "ge",
        description="Transition to perform the experiment on. May be any"
        " transition supported by the quantum operations.",
    )
    do_pca: bool = workflow.option_field(
        False,
        description="Whether to perform principal component analysis on the raw data"
        " independent of whether there were calibration traces in the experiment.",
    )
    use_cal_traces: bool = workflow.option_field(
        True, description="Whether to include calibration traces in the experiment."
    )
