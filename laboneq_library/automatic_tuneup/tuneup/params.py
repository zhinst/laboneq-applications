from dataclasses import dataclass
from typing import List

from laboneq.simple import SweepParameter


@dataclass
class SweepParams:
    amplitude: List[SweepParameter] | SweepParameter = None
    frequency: List[SweepParameter] | SweepParameter = None
    delay: List[SweepParameter] | SweepParameter = None
    flux: SweepParameter = None

    def __post_init__(self):
        # if amplitude is not None and not a list, convert it to a list
        # do the same for other parameters
        if self.amplitude is not None and not isinstance(self.amplitude, list):
            self.amplitude = [self.amplitude]
        if self.frequency is not None and not isinstance(self.frequency, list):
            self.frequency = [self.frequency]
        if self.delay is not None and not isinstance(self.delay, list):
            self.delay = [self.delay]
