# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Serializer for uncertainties package."""

from __future__ import annotations

import uncertainties as unc
from laboneq.serializers.base import VersionedClassSerializer
from laboneq.serializers.serializer_registry import serializer

# ruff: noqa: ARG003


@serializer(types=[unc.core.Variable], public=True)
class UncertaintiesSerializer(VersionedClassSerializer[unc.core.Variable]):
    """Serializer for uncertainties package.

    NOTE: Uncertainties has JSON format on their roadmap
    """

    SERIALIZER_ID = "laboneq_applications.serializers.UncertaintiesSerializer"
    VERSION = 1

    @classmethod
    def to_dict(
        cls,
        obj: unc.core.Variable,
        *args: object,
        **kwargs: object,
    ) -> object:
        """To dict."""
        return {
            "__serializer__": cls.serializer_id(),
            "__version__": cls.version(),
            "__data__": {
                "value": obj.nominal_value,
                "std_dev": obj.std_dev,
                # uncertainties string representation does not save tag
                "tag": obj.tag,
            },
        }

    @classmethod
    def from_dict_v1(
        cls,
        serialized_data: dict,
        *args: object,
        **kwargs: object,
    ) -> unc.core.Variable:
        """From dict."""
        # unc.ufloat_fromstr drops digits
        return unc.core.Variable(**serialized_data["__data__"])
