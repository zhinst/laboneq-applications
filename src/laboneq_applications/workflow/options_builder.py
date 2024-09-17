"""OptionBuilder class to build options for a workflow."""

from __future__ import annotations

import typing
from collections import UserList

from laboneq_applications.workflow.options import WorkflowOptions

if typing.TYPE_CHECKING:
    from laboneq_applications.workflow.options_base import BaseOptions


class OptionBuilder:
    """A class to build options for a workflow."""

    def __init__(self, base: WorkflowOptions) -> None:
        self._base = base
        self._flatten_opts = _get_all_fields(self._base)

    @property
    def base(self) -> WorkflowOptions:
        """Return the base options."""
        return self._base

    def __dir__(self):
        return sorted(self._flatten_opts)

    def __getattr__(self, name: str) -> OptionInfoList:
        if name.startswith("_"):
            return super().__getattribute__(name)
        if name in self._flatten_opts:
            return _retrieve_option_attributes(self._base, name)
        return super().__getattribute__(name)

    def __setattr__(self, name: str, value: typing.Any) -> None:  # noqa: ANN401
        if name.startswith("_"):
            return super().__setattr__(name, value)
        if name in self._flatten_opts:
            raise TypeError(
                "Setting options by assignment is not allowed. "
                "Please use the method call."
            )
        return super().__setattr__(name, value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OptionBuilder):
            return NotImplemented
        return self._base == other._base


def _retrieve_option_attributes(
    option: WorkflowOptions, field_name: str
) -> OptionInfoList:
    """Retrieve an attribute of an option in `task_options`."""
    option_list = OptionInfoList()
    opt_field = option.model_fields.get(field_name, None)
    if opt_field is not None:
        option_list.append(OptionInfo("base", field_name, option))
    for task_name, opt in option.task_options.items():
        if isinstance(opt, WorkflowOptions):
            temp = _retrieve_option_attributes(opt, field_name)
            for t in temp:
                t.name = task_name + "." + t.name
            option_list.extend(temp)
        elif hasattr(opt, field_name):
            option_list.append(OptionInfo(task_name, field_name, opt))
    return option_list


def _get_all_fields(option: WorkflowOptions) -> set[str]:
    """Return all fields in the task_options and the top level fields."""
    all_fields = set(option.model_fields.keys())
    for opt in option.task_options.values():
        if isinstance(opt, WorkflowOptions):
            all_fields.update(_get_all_fields(opt))
        else:
            for field_name, _ in opt:
                all_fields.add(field_name)
    top_level_options = option.model_fields.keys()
    all_fields.update(top_level_options)
    return all_fields


class OptionInfoList(UserList):
    """A list of option information, used to set or read options."""

    def __init__(self, elements: list[OptionInfo] | OptionInfoList | None = None):
        super().__init__(elements or [])

    def __getitem__(self, item: typing.Any) -> OptionInfo | OptionInfoList:  # noqa: ANN401
        return self.data[item]

    def __call__(self, value: typing.Any) -> None:  # noqa: ANN401
        """Set the value for all options in the list."""
        for node in self:
            node(value)

    def __str__(self):
        return f"[{', '.join(str(item) for item in self)}]"

    def _repr_pretty_(self, p, _cycle):  # noqa: ANN001, ANN202
        p.text(str(self))


class OptionInfo:
    """A class representing an option field in the workflow.

    Arguments:
        task_name: Full path of the option field.
            Examples: nested_wf.task1
            or task1 if task1 is at the base of the option.
            "base" if it is a top-layer option field.
        field: Name of the option field.
        option: The option instance.
    """

    def __init__(self, task_name: str, field: str, option: BaseOptions) -> None:
        self.name = task_name
        self.field = field
        self.opt = option
        self._value = str(getattr(self.opt, self.field, self.opt))

    def __call__(self, value: typing.Any) -> None:  # noqa: ANN401
        """Set the value of the option."""
        setattr(self.opt, self.field, value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OptionInfo):
            return NotImplemented
        return (
            self.name == other.name
            and self.field == other.field
            and self.opt == other.opt
            and self._value == other._value
        )

    def __str__(self) -> str:
        return f"({self.name},{self._value})"

    def _repr_pretty_(self, p, _cycle):  # noqa: ANN001, ANN202
        p.text(str(self))
