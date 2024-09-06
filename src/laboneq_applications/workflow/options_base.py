"""Base options for workflow."""

from __future__ import annotations

from io import StringIO
from typing import final

from pydantic import BaseModel, ConfigDict, model_validator
from pydantic_core import PydanticUndefinedType
from rich.console import Console
from rich.pretty import pprint


class BaseOptions(BaseModel):
    """Base class for all Option classes."""

    model_config = ConfigDict(
        validate_assignment=True,
        validate_default=True,
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def _check_defaults(cls, data):  # noqa: ANN206, ANN001
        for name, field in cls.model_fields.items():
            if (
                isinstance(field.default, PydanticUndefinedType)
                and field.default_factory is None
            ):
                raise ValueError(f"Field {name!r} does not have a default value")
        return data

    def to_dict(self) -> dict:
        """Generate a dictionary representation of the options."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> BaseOptions:
        """Create an instance of the class from a dictionary."""
        return cls(**data)

    def __eq__(self, other: BaseOptions) -> bool:
        """Check if two options are equal."""
        if not isinstance(other, BaseOptions):
            return False
        return self.to_dict() == other.to_dict()

    @final
    def __str__(self):
        with StringIO() as buffer:
            console = Console(file=buffer)
            pprint(self, console=console, expand_all=True, indent_guides=True)
            return buffer.getvalue()

    @final
    def __format__(self, _):  # noqa: ANN001
        return self.__repr__()

    @final
    def _repr_pretty_(self, p, _cycle):  # noqa: ANN001, ANN202
        # For Notebooks
        p.text(str(self))
