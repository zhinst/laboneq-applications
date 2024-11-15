# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Tests for uncertainties serializer."""

import json

import pytest
import uncertainties as unc
from laboneq import serializers

from laboneq_applications.serializers import UncertaintiesSerializer


@pytest.mark.parametrize(
    "value",
    [unc.ufloat(1.1, 0.32), unc.ufloat(1, 2, "tag"), unc.ufloat(0.123, 0.321, "tag")],
)
def test_ufloat(value):
    ser = json.dumps(UncertaintiesSerializer.to_dict(value))
    deser = UncertaintiesSerializer.from_dict_v1(json.loads(ser))
    # Variable eq does not work
    assert deser.nominal_value == value.nominal_value
    assert deser.std_dev == value.std_dev
    assert deser.tag == value.tag
    assert deser.std_score(1) == value.std_score(1)


@pytest.mark.parametrize(
    "value",
    [unc.ufloat(1.1, 0.32), unc.ufloat(1, 2, "tag"), unc.ufloat(0.123, 0.321, "tag")],
)
def test_laboneq_json(value):
    ser = serializers.to_json(value)
    deser = serializers.from_json(ser)
    # Variable eq does not work
    assert deser.nominal_value == value.nominal_value
    assert deser.std_dev == value.std_dev
    assert deser.tag == value.tag
    assert deser.std_score(1) == value.std_score(1)
