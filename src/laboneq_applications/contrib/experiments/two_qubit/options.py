# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Options for the contrib two-qubit experiment workflows."""

from __future__ import annotations

from laboneq import workflow

from laboneq_applications.experiments.options import (
    TuneupExperimentOptions,
)


@workflow.task_options(base_class=TuneupExperimentOptions)
class TuneupExperiment2QOptions:
    """Options for the two-qubit tune-up experiments.

    See [TuneupExperimentOptions][] for additional accepted options.

    Attributes:
        auto_chunking:
            Whether to let the compiler split the sweep into chunks.
            Default: `False`.
    """

    auto_chunking: bool = workflow.option_field(
        default=False,
        description="Whether to let the compiler split the sweep into chunks.",
    )
