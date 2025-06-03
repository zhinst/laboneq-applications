# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""LabOne Q Application Library provides some utilities to verify the experiments.

The `CompiledExperimentVerifier` class is used to verify the workflow results.
It provides methods to assert the number of pulses and pulse attributes such as
pulse timing and pulse parameterization.

## Example

```python
from laboneq_applications.experiments import amplitude_rabi

result = amplitude_rabi(session, quantum_operations, qubits, amplitudes)
compiled_experiment = result.tasks["compile_experiment"].output
verifier = CompiledExperimentVerifier(compiled_experiment, max_events=5000)
verifier.assert_number_of_pulses("/logical_signal_groups/q0/drive", 8)
verifier.assert_pulse(
    signal = "/logical_signal_groups/q0/drive",
    index = 3,
    start = 12.168e-6,
    stop = 12.219e-6,
    parameterized_with = [],
)
```

"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from laboneq.pulse_sheet_viewer.pulse_sheet_viewer import (
    _fill_maybe_missing_information,
)

if TYPE_CHECKING:
    from laboneq.simple import CompiledExperiment


@dataclass
class _SignalEvent:
    start: list = field(default_factory=list)
    end: list = field(default_factory=list)


@dataclass
class _Pulse:
    """Class to represent the pulse.

    More to be added in the future: amplitude, phase,
    frequency(only for measure/acquire pulses)
    """

    start: float
    end: float
    parameterized_with: list[str]


class CompiledExperimentVerifier:
    """Class to verify the compiled experiment.

    Arguments:
        compiled_experiment:
            The L1Q compiled experiment to verify.
        max_events:
            The maximum number of events to analyze.
            Default is 5000.
    """

    _TIMING_TOLERANCE = 1e-12

    def __init__(
        self,
        compiled_experiment: CompiledExperiment,
        max_events: int = 5000,
    ) -> None:
        self.pulse_extractor = _PulseExtractorPSV(
            compiled_experiment,
            max_events=max_events,
        )

    def assert_number_of_pulses(self, signal: str, pulse_number: int) -> None:
        """Assert the number of pulses played for a particular signal.

        Arguments:
            signal:
                The signal for which the number of pulses is required.
            pulse_number:
                The expected number of pulses.
        """
        actual_pulse_number = self.pulse_extractor.get_pulse_count(signal)
        err_message = (
            f"Number of pulses mismatch for signal {signal} "
            f"expected {actual_pulse_number} got {pulse_number}"
        )
        if actual_pulse_number != pulse_number:
            raise AssertionError(err_message)

    def assert_pulse(
        self,
        signal: str,
        index: int,
        start: float | None = None,
        end: float | None = None,
        length: float | None = None,
        parameterized_with: list[str] | None = None,
        tolerance: float = _TIMING_TOLERANCE,
    ) -> None:
        """Assert the properties of a particular pulse played.

        Arguments:
            signal:
                The signal name for which the pulse is required.
            index:
                The 0-based index of pulse. Counting only within the signal.
            start:
                The expected start time of the pulse.
                If None, the start time is not checked.
            end:
                The expected end time of the pulse.
                If None, the end time is not checked.
            length:
                The expected length of the pulse.
                If None, the length is not checked.
            parameterized_with:
                The parameter names that parameterize the pulse.
                If None, the parameterized_with is not checked.
            tolerance:
                The tolerance for the comparison.
                If not provided, the default tolerance of 1e-12 is used.
        """
        pulse = self.pulse_extractor.get_pulse(signal, index)
        if start is not None:
            np.testing.assert_allclose(
                start,
                pulse.start,
                atol=tolerance,
                err_msg="Start time mismatch",
            )
        if end is not None:
            np.testing.assert_allclose(
                end,
                pulse.end,
                atol=tolerance,
                err_msg="End time mismatch",
            )
        if length is not None:
            np.testing.assert_allclose(
                length,
                pulse.end - pulse.start,
                atol=tolerance,
                err_msg="Pulse length mismatch",
            )
        if (
            parameterized_with is not None
            and pulse.parameterized_with != parameterized_with
        ):
            raise AssertionError(
                f"Parameterized with mismatch, expected {parameterized_with} "
                f"got {pulse.parameterized_with}",
            )

    def assert_pulse_pair(
        self,
        signals: tuple[str, str] | str,
        indices: tuple[int, int],
        start: float | None = None,
        end: float | None = None,
        distance: float | None = None,
        tolerance: float = _TIMING_TOLERANCE,
    ) -> None:
        """Assert the properties of a pair of pulses played.

        Arguments:
            signals:
                The tuple of signals for which the pulse pair is required.
                If a single signal is provided, the same signal is used for both pulses.
            indices:
                The tuple of 0-based indices of pulses.
            start:
                The expected start time of the pulse pair.
                If None, the start time is not checked.
            end:
                The expected end time of the pulse pair.
                If None, the end time is not checked.
            distance:
                The difference of the start time of the second pulse
                and the end time of the first pulse.
                If None, the difference is not checked.
            tolerance:
                The tolerance for the comparison.
                If not provided, the default tolerance of 1e-12 is used.
        """
        if len(indices) != 2:  # noqa: PLR2004
            raise ValueError("Indices must be a tuple of two integers")
        if isinstance(signals, str):
            signals = (signals, signals)
        signal_0, signal_1 = signals
        pulse_0 = self.pulse_extractor.get_pulse(signal_0, indices[0])
        pulse_1 = self.pulse_extractor.get_pulse(signal_1, indices[1])

        if start is not None:
            actual_start = pulse_1.start - pulse_0.start
            np.testing.assert_allclose(start, actual_start, atol=tolerance)

        if end is not None:
            actual_end = pulse_1.end - pulse_0.end
            np.testing.assert_allclose(end, actual_end, atol=tolerance)

        if distance is not None:
            actual_distance = pulse_1.start - pulse_0.end
            np.testing.assert_allclose(distance, actual_distance, atol=tolerance)


class _PulseExtractorPSV:
    """Class to verify the compiled experiment.

    Attributes:
        compiled_experiment:
            L1Q compiled experiment.
        max_events:
            The maximum number of events to analyze.
    """

    def __init__(
        self,
        compiled_experiment: CompiledExperiment,
        max_events: int = 5000,
    ) -> None:
        self._max_events = max_events
        self._compiled_experiment = compiled_experiment
        self._process_compiled_exp()

    @property
    def max_events(self) -> int:
        return self._max_events

    def _process_compiled_exp(self) -> None:
        compiled_experiment = _fill_maybe_missing_information(
            self._compiled_experiment,
            max_events_to_publish=self._max_events,
        )
        # _fill_maybe_missing_information re-compiles experiments with
        # max_events_to_publish and output_extras=True to produce the
        # event_list
        event_list = compiled_experiment.scheduled_experiment.schedule["event_list"]

        self._event_collection = self._process_event_list(event_list)

    def _process_event_list(self, event_list: list) -> dict[str, _SignalEvent]:
        start_events = ("PLAY_START", "ACQUIRE_START")
        end_events = ("PLAY_END", "ACQUIRE_END")
        event_collection = defaultdict(_SignalEvent)
        for event in event_list:
            if event["event_type"] in set(start_events + end_events):
                signal_name = event["signal"]
                if event["event_type"] in start_events:
                    event_collection[signal_name].start.append(event)
                elif event["event_type"] in end_events:
                    event_collection[signal_name].end.append(event)
        return event_collection

    def _is_equal_length(self, signal: str) -> bool:
        return len(self._event_collection[signal].start) == len(
            self._event_collection[signal].end,
        )

    def _check_pulse_number(self, pulse_number: int, signal: str) -> None:
        if pulse_number > self.get_pulse_count(signal):
            raise ValueError(
                f"Pulse number out of range, max pulse number is "
                f"{self.get_pulse_count(signal)}",
            )

    def get_pulse_count(
        self,
        signal: str,
    ) -> int:
        """Get the number of pulses played for a particular signal.

        Arguments:
            signal:
                The signal for which the number of pulses is required.

        Returns:
            The number of pulses played for the signal.

        Raises:
            ValueError:
                If the number of START and END events in the signal are not equal.
        """
        if not self._is_equal_length(signal):
            raise ValueError(
                f"Mismatch in number of START and END events of {signal}",
                "Consider increase max_events_to_publish, currently",
                f"set to {self.max_events}",
            )
        return len(self._event_collection[signal].end)

    def get_pulse(self, signal: str, index: int) -> _Pulse:
        """Get the pulse information.

        Arguments:
            signal:
                The signal for which the pulse information is required.
            index:
                The 0-based index for pulse.

        Returns:
            The _Pulse object containing information about the pulse.

        Raises:
            ValueError:
                If the index is out of range.
        """
        self._check_pulse_number(index, signal)
        start = self._event_collection[signal].start[index]["time"]
        end = self._event_collection[signal].end[index]["time"]
        parameterized_with = self._event_collection[signal].start[index][
            "parametrized_with"
        ]
        return _Pulse(start, end, parameterized_with)
