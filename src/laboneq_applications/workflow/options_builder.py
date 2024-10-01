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

    def __getattr__(self, field_name: str) -> OptionNodeList:
        if field_name.startswith("_"):
            return super().__getattribute__(field_name)
        if field_name in self._flatten_opts:
            return _retrieve_option_attributes(self._base, field_name)
        return super().__getattribute__(field_name)

    def __setattr__(self, field_name: str, value: typing.Any) -> None:  # noqa: ANN401
        if field_name.startswith("_"):
            return super().__setattr__(field_name, value)
        if field_name in self._flatten_opts:
            raise TypeError(
                "Setting options by assignment is not allowed. "
                "Please use the method call."
            )
        return super().__setattr__(field_name, value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OptionBuilder):
            return NotImplemented
        return self._base == other._base


def _retrieve_option_attributes(
    option: WorkflowOptions,
    field_name: str,
    current_node: str = "base",
) -> OptionNodeList:
    """Return OptionNodeList that contains the option fields."""
    option_list = OptionNodeList()
    opt_field = option.model_fields.get(field_name, None)
    if opt_field is not None:
        option_list.append(OptionNode(current_node, field_name, option))
    for task_name, opt in option._task_options.items():
        new_node = current_node + "." + task_name
        if isinstance(opt, WorkflowOptions):
            temp = _retrieve_option_attributes(opt, field_name, new_node)
            option_list.extend(temp)
        elif hasattr(opt, field_name):
            option_list.append(OptionNode(new_node, field_name, opt))
    return option_list


def _get_all_fields(option: WorkflowOptions) -> set[str]:
    """Return all fields in the task_options and the top level fields."""
    all_fields = set(option.model_fields.keys())
    for opt in option._task_options.values():
        if isinstance(opt, WorkflowOptions):
            all_fields.update(_get_all_fields(opt))
        else:
            for field_name, _ in opt:
                all_fields.add(field_name)
    top_level_options = option.model_fields.keys()
    all_fields.update(top_level_options)
    return all_fields


class OptionNodeList(UserList):
    """A list of option nodes for setting and querying options values."""

    def __init__(self, elements: list[OptionNode] | OptionNodeList | None = None):
        super().__init__(elements or [])

    def __getitem__(self, item: typing.Any) -> OptionNode | OptionNodeList:  # noqa: ANN401
        if isinstance(item, slice):
            return type(self)(self.data[item])
        return self.data[item]

    def __call__(self, value: typing.Any, selector: str | None = None) -> None:  # noqa: ANN401
        """Set the value of option fields, selected by a path name of a task/workflow.

        Arguments:
            value: The value to set.
            selector: Path of the task or workflow.
                If None, set the value for all option fields in the list.

        Example:
            ```python
            opt = workflow.options()

            # To set the value of field `count` at the top-level
            opt.count(1, ".")

            # To set a value to field `count` of a sub-workflow at top-level
            opt.count(1, "sub_workflow")

            # a task at top-level
            opt.count(1, "task1)

            # a task nested in `sub_workflow`
            opt.count(1, "sub_workflow.task2")

            # Or a workflow nested in another workflow
            opt.count(1, "sub_workflow.nested_workflow")
            ```
        """
        filtered = self if selector is None else self._get_filtered(selector)
        for element in filtered:
            element(value)

    def _get_filtered(self, task_name: str) -> OptionNodeList:
        # OptionNode is represented using "full-path" format, aka base.wf1.task1
        # But we'd like to omit "base" when setting the fields; "wf1.task1"
        # would be sufficient.
        filter_name = "base." + task_name if task_name != "." else task_name
        filtered = OptionNodeList(
            [node for node in self if self._predicate(node, filter_name)]
        )
        if not filtered:
            raise ValueError(
                f"Task or workflow {task_name} not found to have the option "
                f"{self[0].field}."
            )
        return filtered

    def _predicate(self, item: OptionNode, name: str) -> bool:
        if name == ".":
            return item.is_top_level()
        return name == item.name or name == item.name.rsplit(".", 1)[0]

    def __str__(self):
        return f"[{', '.join(str(item) for item in self)}]"

    def _repr_pretty_(self, p, _cycle):  # noqa: ANN001, ANN202
        p.text(str(self))


class OptionNode:
    """A class representing a node for the options in workflow.

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

    def is_top_level(self) -> bool:
        """Return True if the option is a top-level field."""
        splitted = self.name.split(".")
        return splitted[0] == "base" and len(splitted) <= 2  # noqa: PLR2004

    def __call__(self, value: typing.Any) -> None:  # noqa: ANN401
        """Set the value of the option."""
        setattr(self.opt, self.field, value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OptionNode):
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
