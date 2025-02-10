# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Experiment task and workflow options."""

from __future__ import annotations

from typing import Literal, TypeVar

import attrs
from laboneq.simple import AcquisitionType, AveragingMode, RepetitionMode
from laboneq.workflow import (
    option_field,
    task_options,
    workflow_options,
)

T = TypeVar("T")


def _parse_acquisition_type(v: str | AcquisitionType) -> AcquisitionType:
    return AcquisitionType(v)


def _parse_averaging_mode(v: str | AveragingMode) -> AveragingMode:
    return AveragingMode(v)


def _parse_repetition_mode(v: str | RepetitionMode) -> RepetitionMode:
    return RepetitionMode(v)


@task_options
class BaseExperimentOptions:
    """Base options for the experiment.

    Attributes:
        count:
            The number of repetitions.
            Default: A common choice in practice, 1024.
        averaging_mode:
            Averaging mode to use for the experiment.
            Default: `AveragingMode.CYCLIC`.
        acquisition_type:
            Acquisition type to use for the experiment.
            Default: `AcquisitionType.INTEGRATION`.
        repetition_mode:
            The repetition mode to use for the experiment.
            Default: `RepetitionMode.FASTEST`.
        repetition_time:
            The repetition time.
            Default: None.
        reset_oscillator_phase:
            Whether to reset the oscillator phase.
            Default: False.
        active_reset (bool):
            Whether to use active reset.
            Default: False.
        active_reset_repetitions (int):
            The number of times to repeat the active resets.
            Default: 1
        active_reset_states (str | tuple | None):
            The qubit states to actively reset.
            Default: "ge"
    """

    count: int = option_field(default=1024, description="The number of repetitions.")
    acquisition_type: str | AcquisitionType = option_field(
        AcquisitionType.INTEGRATION,
        description="Acquisition type to use for the experiment.",
        converter=_parse_acquisition_type,
    )
    averaging_mode: str | AveragingMode = option_field(
        AveragingMode.CYCLIC,
        description="Averaging mode to use for the experiment.",
        converter=_parse_averaging_mode,
    )
    repetition_mode: str | RepetitionMode = option_field(
        RepetitionMode.FASTEST,
        description="The repetition mode to use for the experiment.",
        converter=_parse_repetition_mode,
    )
    repetition_time: float | None = option_field(
        None, description="The repetition time."
    )
    reset_oscillator_phase: bool = option_field(
        False, description="Whether to reset the oscillator phase."
    )
    active_reset: bool = option_field(False, description="Whether to use active reset.")
    active_reset_repetitions: int = option_field(
        1, description="The number of times to repeat the active resets."
    )
    active_reset_states: str | tuple | None = option_field(
        "ge", description="The qubit states to actively reset."
    )


@task_options(base_class=BaseExperimentOptions)
class TuneupExperimentOptions:
    """Base options for a tune-up experiment.

    Attributes:
        transition:
            Transition to perform the experiment on. May be any
            transition supported by the quantum operations.
            Default: `"ge"` (i.e. ground to first excited state).
        use_cal_traces:
            Whether to include calibration traces in the experiment.
            Default: `True`.
        cal_states:
            The states to prepare in the calibration traces. Can be any
            string or tuple made from combining the characters 'g', 'e', 'f'.
            Default: same as transition
    """

    transition: Literal["ge", "ef"] = option_field(
        "ge",
        description="Transition to perform the experiment on. May be any"
        " transition supported by the quantum operations.",
    )
    use_cal_traces: bool = option_field(
        True, description="Whether to include calibration traces in the experiment."
    )
    cal_states: str | tuple = option_field(
        "ge", description="The states to prepare in the calibration traces."
    )


# create additional options for spectroscopy
@task_options(base_class=BaseExperimentOptions)
class ResonatorSpectroscopyExperimentOptions:
    """Base options for the resonator spectroscopy experiment.

    Additional attributes:
        use_cw:
            Perform a CW spectroscopy where no measure pulse is played.
            Default: False.
        spectroscopy_reset_delay:
            How long to wait after an acquisition in seconds.
            Default: 1e-6.
        acquisition_type:
            Acquisition type to use for the experiment.
            Default: `AcquisitionType.SPECTROSCOPY`.
    """

    use_cw: bool = option_field(
        False, description="Perform a CW spectroscopy where no measure pulse is played."
    )
    spectroscopy_reset_delay: float = option_field(
        1e-6, description="How long to wait after an acquisition in seconds."
    )
    acquisition_type: AcquisitionType = option_field(
        AcquisitionType.SPECTROSCOPY,
        description="Acquisition type to use for the experiment.",
    )


@workflow_options
class TuneUpWorkflowOptions:
    """Option class for tune-up experiment workflows.

    Attributes:
        do_analysis (bool):
            Whether to run the analysis workflow.
            Default: True
        update (bool):
            Whether to update the setup based on the results from the analysis.
            Default: False
    """

    do_analysis: bool = option_field(
        True, description="Whether to run the analysis workflow."
    )
    update: bool = option_field(
        False,
        description="Whether to update the setup based on the "
        "results from the analysis.",
    )


@task_options(base_class=BaseExperimentOptions)
class QubitSpectroscopyExperimentOptions:
    """Base options for the resonator spectroscopy experiment.

    Additional attributes:
        spectroscopy_reset_delay:
            How long to wait after an acquisition in seconds.
            Default: 1e-6.
    """

    spectroscopy_reset_delay: float = option_field(
        1e-6, description="How long to wait after an acquisition in seconds."
    )


@task_options(base_class=BaseExperimentOptions)
class TWPASpectroscopyExperimentOptions:
    """Base options for the TWPA spectroscopy experiment.

    Additional attributes:
        use_probe_from_ppc:
            Use the probe tone from the SHFPPC instead of the QA out.
            Default: False.
        use_cw:
            Perform a CW spectroscopy where no measure pulse is played.
            Default: False.
        spectroscopy_reset_delay:
            How long to wait after an acquisition in seconds.
            Default: 1e-6.
        acquisition_type:
            Acquisition type to use for the experiment.
            Default: `AcquisitionType.SPECTROSCOPY`.

    Raises:
        ValueError:
            If the acquisition_type is not AcquisitionType.SPECTROSCOPY
            or AcquisitionType.SPECTROSCOPY_PSD.
    """

    use_probe_from_ppc: bool = option_field(
        False, description="Use the probe tone from the SHFPPC instead of the QA out."
    )
    use_cw: bool = option_field(
        False, description="Perform a CW spectroscopy where no measure pulse is played."
    )
    spectroscopy_reset_delay: float = option_field(
        1e-6, description="How long to wait after an acquisition in seconds."
    )
    acquisition_type: AcquisitionType = option_field(
        AcquisitionType.SPECTROSCOPY,
        description="Acquisition type to use for the experiment.",
        converter=_parse_acquisition_type,
        validators=attrs.validators.in_(
            [AcquisitionType.SPECTROSCOPY, AcquisitionType.SPECTROSCOPY_PSD]
        ),
    )


@task_options(base_class=BaseExperimentOptions)
class TWPATuneUpExperimentOptions:
    """Base options for TWPA Tuneup experiments.

    Additional attributes:
        use_probe_from_ppc:
            Use the probe tone from the SHFPPC instead of the QA out.
            Default: False.
        spectroscopy_reset_delay:
            How long to wait after an acquisition in seconds.
            Default: 1e-6.
        acquisition_type:
            Acquisition type to use for the experiment.
            Default: `AcquisitionType.SPECTROSCOPY`.
        averaging_mode:
            Averaging mode to use for the experiment.
            Default: `AveragingMode.SEQUENTIAL`.
        do_snr (bool):
            Whether to run SNR measurement.

    Raises:
        ValueError:
            If the acquisition_type is not AcquisitionType.SPECTROSCOPY
            or AcquisitionType.SPECTROSCOPY_PSD.
    """

    use_probe_from_ppc: bool = option_field(
        False, description="Use the probe tone from the SHFPPC instead of the QA out."
    )

    spectroscopy_reset_delay: float = option_field(
        1e-6, description="How long to wait after an acquisition in seconds."
    )
    acquisition_type: AcquisitionType = option_field(
        AcquisitionType.SPECTROSCOPY,
        description="Acquisition type to use for the experiment.",
        converter=_parse_acquisition_type,
        validators=attrs.validators.in_(
            [AcquisitionType.SPECTROSCOPY, AcquisitionType.SPECTROSCOPY_PSD]
        ),
    )
    averaging_mode: str | AveragingMode = option_field(
        AveragingMode.SEQUENTIAL,
        description="Averaging mode to use for the experiment.",
        converter=_parse_averaging_mode,
    )
    do_snr: bool = option_field(False, description="Whether to run SNR measurement.")


@workflow_options(base_class=TuneUpWorkflowOptions)
class TWPATuneUpWorkflowOptions:
    """Option class for tune-up experiment workflows.

    Attributes:
        do_snr (bool):
            Whether to run SNR measurement.
    """

    do_snr: bool = option_field(False, description="Whether to run SNR measurement.")
