# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Test helpers for working with DSL outputs and UIDs."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import pytest
from laboneq.dsl.experiment import (
    Acquire,
    Call,
    Delay,
    Experiment,
    Operation,
    PlayPulse,
    Reserve,
    builtins,
)
from laboneq.dsl.experiment.pulse import PulseFunctional
from laboneq.simple import (
    AcquireLoopRt,
    Calibration,
    Case,
    Match,
    Section,
    SignalCalibration,
    Sweep,
    SweepParameter,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


@pytest.fixture(autouse=True)
def reset_uids():
    """Reset the global UIDs."""
    builtins.reset_global_uid_generator()
    return "This is the reset_uids fixture."


class ExpectedDSLStructure:
    """Helper to hold types and attributes to assert on a LabOne Q DSL tree.

    Arguments:
        type_:
            The expected type of the DSL object. Supported types are
            `Experiment`, `Section`, `Operation`, `PulseFunctional`
            and their subclasses.
        attrs:
            A dictionary of the expected attributes of the DSL
            object and their expected values. The values
            may themselves be `ExpectedDSLStructure` objects
            if, for example, a pulse is expected.
        children:
            An ordered iterable of `ExpectedDSLStructure` instances
            for the children of the DSL object.
    """

    def __init__(
        self,
        type_,
        attrs: dict,
        children: Iterable[ExpectedDSLStructure] = (),
    ):
        self.expected_type = type_
        self.expected_attrs = attrs
        self.expected_children = list(children)

    def __repr__(self):
        if self.expected_attrs:
            attrs = " " + " ".join(f"{k}={v!r}" for k, v in self.expected_attrs.items())
        else:
            attrs = ""
        if self.expected_children:
            children = "".join(
                [
                    " children=[",
                    ", ".join(repr(c) for c in self.expected_children),
                    "]",
                ],
            )
        else:
            children = ""
        return (
            f"<{self.__class__.__name__}."
            f"{self.expected_type.__name__}"
            f"{attrs}{children}>"
        )

    def __eq__(self, other):
        if type(other) is not self.expected_type:
            return False

        children = self._dsl_children(other)
        if len(children) != len(self.expected_children):
            return False

        for k, v in self.expected_attrs.items():
            # v may be an instance of ExpectedDSLStructure
            if not hasattr(other, k):
                return False
            obj_attr = getattr(other, k)
            if not self._careful_equals(obj_attr, v):
                return False

        for child, expected_child in zip(children, self.expected_children):
            if child != expected_child:
                return False

        return True

    def _dsl_children(self, obj):
        """Return the children of the given DSL object."""
        if isinstance(obj, Experiment):
            return obj.sections
        if isinstance(obj, Section):
            return obj.children
        if isinstance(
            obj,
            (
                Calibration,
                Operation,
                PulseFunctional,
                SignalCalibration,
                SweepParameter,
            ),
        ):
            return []
        raise ValueError(f"Unsupported DSL object: {obj!r}")

    def _careful_equals(self, obj_attr, v):
        """Carefully compare two objects avoiding common pitfalls.

        Common exceptional cases addressed by this method:

          - numpy arrays.
        """
        if isinstance(obj_attr, np.ndarray):
            if obj_attr.shape != getattr(v, "shape", None):
                return False
            return all(obj_attr == v)
        if isinstance(obj_attr, (list, tuple)):
            if len(obj_attr) != len(v):
                return False
            return all(self._careful_equals(a, b) for a, b in zip(obj_attr, v))
        if isinstance(obj_attr, SweepParameter):
            # TODO: Remove this work around once the __eq__ methods of the
            #       sweep parameter classes are fixed in laboneq.
            return v == obj_attr
        return obj_attr == v

    def children(self, *args):
        """Add the listed arguments as expected sections or subsections.

        Arguments:
            args:
                Positional arguments to add as children. Each argument
                should be either an instance of `ExpectedDSLStructure` or
                a list of such instances.

        Returns `self` so that children may be added to `ExpectedDSLStructure`
        instances being built inside lists:

        ```
        import tests.helpers.dsl as tsl

        assert section == tsl.section(
            uid="section_0",
        ).children(
            tsl.section(uid="section_1").children(
                tsl.delay_op(time=5.0),
            ),
        )
        ```
        """
        for arg in args:
            if isinstance(arg, self.__class__):
                self.expected_children.append(arg)
            elif isinstance(arg, (list, tuple)):
                self.children(*arg)
            else:
                raise TypeError("Unsupported object passed as children: {arg!r}")
        return self

    def compare(self, obj, indent=0):  # noqa: C901
        """Return a list of strings describing how the given object fails to match.

        Returns:
            A list of strings.
        """
        diff = []

        if type(obj) is not self.expected_type:
            diff.append(
                f"Type: {type(obj).__name__} is not a {self.expected_type.__name__}",
            )

        children = self._dsl_children(obj)
        if len(children) > len(self.expected_children):
            diff.append("Unexpected extra children:")
            diff.extend(
                f"  {type(child).__name__}: {getattr(child, 'uid', '???')}"
                for child in children[len(self.expected_children) :]
            )
        if len(self.expected_children) > len(children):
            diff.append("Missing expected children:")
            diff.extend(
                f"  {expected_child.expected_type.__name__}:"
                f" {expected_child.expected_attrs.get('uid', '???')}"
                for expected_child in self.expected_children[len(children) :]
            )

        for k, v in self.expected_attrs.items():
            if not hasattr(obj, k):
                diff.append(f".{k}: MISSING != {v!r}")
            else:
                obj_attr = getattr(obj, k)
                if isinstance(v, ExpectedDSLStructure):
                    diff.extend(v.compare(obj_attr, indent + 1))
                elif not self._careful_equals(obj_attr, v):
                    diff.append(f".{k}: {obj_attr!r} != {v!r}")

        for child, expected_child in zip(children, self.expected_children):
            diff.extend(expected_child.compare(child, indent + 1))

        obj_name = f"{type(obj).__name__}: {getattr(obj, 'uid', '???')}"

        if diff:
            # label and indent diff if there are differences:
            padding = "  " * indent
            diff = [f"{obj_name}"] + [f"{padding}{line}" for line in diff]
        elif indent == 0:
            # if the top level diff is empty, add a line saying the object matches
            diff = [f"{obj_name} appears to match."]

        return diff


# Experiments


def experiment(**kw):
    """Description of expected experiment and attributes."""
    return ExpectedDSLStructure(Experiment, kw)


# Sections


def section_like(type_, **kw):
    """Description of expected section type and attributes."""
    return ExpectedDSLStructure(type_, kw)


section = partial(section_like, type_=Section)
acquire_loop_rt = partial(section_like, type_=AcquireLoopRt)
sweep = partial(section_like, type_=Sweep)
match = partial(section_like, type_=Match)
case = partial(section_like, type_=Case)


# Operations


def op_like(type_, **kw):
    """Description of expected operation type and attributes."""
    return ExpectedDSLStructure(type_, kw)


acquire_op = partial(op_like, type_=Acquire)
call_op = partial(op_like, type_=Call)
delay_op = partial(op_like, type_=Delay)
play_pulse_op = partial(op_like, type_=PlayPulse)
reserve_op = partial(op_like, type_=Reserve)

# Pulses


def pulse(**kw):
    """Description of expected pulse attributes."""
    return ExpectedDSLStructure(PulseFunctional, kw)


# Parameters


def sweep_parameter(**kw):
    """Description of expected sweep parameter attributes."""
    return ExpectedDSLStructure(SweepParameter, kw)


# Calibration


def calibration(**kw):
    """Description of expected calibration attributes."""
    return ExpectedDSLStructure(Calibration, kw)


def signal_calibration(**kw):
    """Description of expected signal calibration attributes."""
    return ExpectedDSLStructure(SignalCalibration, kw)
