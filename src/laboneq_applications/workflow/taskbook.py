"""This module defines objects related to taskbook.

A taskbook is a collection of tasks and their records.

Whenever a task is executed within a taskbook, its'
arguments, result and other relevant information is
saved into the taskbook records.

## Example

### Building and running the taskbook

```python
from laboneq.workflow import task, taskbook

def addition(x, y):
    return x + y

@task
def my_task(x):
   return x + 1

@taskbook
def my_taskbook(x):
    x = addition(1, x)
    my_task(x)
    return "success"

result = my_taskbook(x=5)
```

### Inspecting the results

The results of all tasks are recorded and may be inspected later
using the result object returned by the taskbook.

The taskbook output and the information of each task execution
can be accessed from the result object:

```python
>>> result.output
"success"
>>> result.tasks[0].output
7
>>> result.tasks[0].args
(6,)
>>> result.tasks["addition"]
Task(name="addition")
>>> taskbook.tasks["addition", :]
[Task(name="addition")]
>>> taskbook.tasks.unique()
["addition", "my_task"]
```

As `addition()` was not marked as a task, it has no records in the taskbook.

### Running functions as tasks

If a function is a normal Python function and you'd wish to run it as a task within the
taskbook without adding the `task()` decorator, it can be done by wrapping the
function with it.

This also works if you wish to save only specific calls to the function.

```python
@taskbook
def my_taskbook(x):
    x = task(addition)(1, x)  # Record will be saved
    y = addition(2, x)  # No record saved
```

Now the normal Python function `addition()` wrapped in `task()` will have records
within the taskbook.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from functools import wraps
from typing import TYPE_CHECKING, overload

from laboneq_applications.workflow._context import (
    ExecutorContext,
    LocalContext,
    get_active_context,
)

if TYPE_CHECKING:
    from typing import Callable

    from laboneq_applications.workflow.task import task_


class Task:
    """A task.

    The instance holds execution information of an task when it
    was executed in a taskbook.

    Attributes:
        name: The name of the task.
        func: Underlying Python function.
        output: The output of the function.
        args: Arguments of the task.
        kwargs: Keyword arguments of the task.
    """

    def __init__(
        self,
        task: task_,
        output: object,
        args: object,
        kwargs: object,
    ) -> None:
        self._task = task
        self._output = output
        self.args = args
        self.kwargs = kwargs

    @property
    def name(self) -> str:
        """Task name."""
        return self._task.name

    @property
    def func(self) -> Callable:
        """Underlying function."""
        return self._task.func

    @property
    def src(self) -> str:
        """Source code of the task."""
        return self._task.src

    @property
    def output(self) -> object:
        """Output of the task."""
        return self._output

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Task):
            return NotImplemented
        return (
            self._task == value._task
            and self.output == value.output
            and self.args == value.args
            and self.kwargs == value.kwargs
        )

    def __repr__(self) -> str:
        return f"Task(name={self.name})"


class TasksView(Sequence):
    """A view of tasks.

    This class provides a view into tasks.

    Arguments:
        tasks: List of tasks.

    The class is a `Sequence` of tasks, however item lookup
    is modified to support the following cases:

        - Lookup by index and slicing
        - Lookup by name (string)
        - Lookup by name and slicing
    """

    def __init__(self, tasks: list[Task] | None = None) -> None:
        self._tasks = tasks or []

    def unique(self) -> set[str]:
        """Return unique names of the tasks."""
        return {t.name for t in self._tasks}

    def __repr__(self) -> str:
        return str(self._tasks)

    @overload
    def __getitem__(self, item: tuple[str, int | slice] | slice) -> list[Task]: ...

    @overload
    def __getitem__(self, item: str | int) -> Task: ...

    def __getitem__(
        self,
        item: str | int | tuple[str, int | slice] | slice,
    ) -> Task | list[Task]:
        """Get a single or multiple tasks.

        Arguments:
            item: Index, name of the task, slice or a tuple.

                If index or name is given, the return value will be a single `Task`,
                the first one found in the sequence.

                tuple: A tuple of format (<name>, <index/slice>) will return
                    list of tasks.

        Returns:
            Task or list of tasks, depending on the input filter.

        Raises:
            KeyError: Task by name was not found.
            IndexError: Task by index was not found.
        """
        if isinstance(item, str):
            try:
                return next(t for t in self._tasks if t.name == item)
            except StopIteration:
                raise KeyError(item) from None
        if isinstance(item, tuple):
            items = [t for t in self._tasks if t.name == item[0]]
            if not items:
                raise KeyError(item[0])
            return items[item[1]]
        return self._tasks[item]

    def __len__(self) -> int:
        return len(self._tasks)

    def __eq__(self, other: object) -> bool:
        return self._tasks == other


class TaskBook:
    """A taskbook.

    A taskbook is a collection of executed tasks and their results.
    """

    def __init__(self, options: dict | None = None) -> None:
        self._tasks: list[Task] = []
        self._output = None
        if options is None:
            self._task_options = TaskBookOptions()
        else:
            self._task_options = TaskBookOptions(**options)

    @property
    def tasks(self) -> TasksView:
        """Task entries of the taskbook.

        The ordering of the tasks is the order of the execution.

        Tasks is a `Sequence` of tasks, however item lookup
        is modified to support the following cases:

        Example:
            ```python
            taskbook = my_taskbook()
            taskbook.tasks["run_experiment"]  # First task of name 'run_experiment'
            taskbook.tasks["run_experiment", :]  # All tasks named 'run_experiment'
            taskbook.tasks["run_experiment", 1:5]  # Slice tasks named 'run_experiment'
            taskbook.tasks[0]  # First executed task
            taskbook.tasks[0:5]  # Slicing

            taskbook.tasks.unique() # Unique task names
            ```
        """
        return TasksView(self._tasks)

    @property
    def task_options(self) -> TaskBookOptions:
        """Task options of the taskbook."""
        return self._task_options

    @property
    def output(self) -> object:
        """Output of the taskbook."""
        return self._output

    def add_entry(self, entry: Task) -> None:
        """Add an entry to the taskbook.

        Arguments:
            entry: Task entry.
        """
        self._tasks.append(entry)

    def __repr__(self):
        return f"TaskBook(tasks={self.tasks})"


class _TaskBookExecutor(ExecutorContext):
    """A taskbook executor."""

    def __init__(self, taskbook: TaskBook) -> None:
        self.taskbook = taskbook

    def execute_task(
        self,
        task: task_,
        *args: object,
        **kwargs: object,
    ) -> None:
        # TODO: Error handling and saving of the exception during execution

        opts = self.taskbook.task_options

        if opts.task_options(task.name) and task.has_opts:
            # if a task is called with options explicitly in the taskbook,
            # update the options with the task options.
            # Otherwise use the task options.
            # If no options are provided at the taskbook level,
            # don't update the options.
            kwargs.setdefault("options", {}).update(opts.task_options(task.name))

        r = task._run(*args, **kwargs)
        entry = Task(
            task=task,
            output=r,
            args=args,
            kwargs=kwargs,
        )
        self.taskbook.add_entry(entry)
        return r


def taskbook(func: Callable) -> Callable[..., TaskBook]:
    """A decorator to turn a function into a taskbook.

    When a function is marked as a taskbook, it will record
    each task's information. Otherwise the taskbook behaves just
    like an ordinary Python function.

    Arguments:
        func: Function to be marked as a taskbook.

    Returns:
        A taskbook which holds the records of each executed task and
            the return value of the taskbook function.
    """

    @wraps(func)
    def inner(*args: object, **kwargs: object) -> TaskBook:
        ctx = get_active_context()
        if isinstance(ctx, _TaskBookExecutor):
            # TODO: Should nested books append to the top level or?
            raise NotImplementedError("Taskbooks cannot be nested.")
        opts = kwargs.get("options", None)
        book = TaskBook(options=opts)
        with LocalContext.scoped(_TaskBookExecutor(book)):
            book._output = func(*args, **kwargs)
        return book

    return inner


class TaskBookOptions:
    """A class for organizing options for the taskbook."""

    _DELIMITER = "."
    _PREFIX = "task"

    def __init__(self, **kwargs) -> None:
        self._broadcast = {}  # contain broadcast options
        self._specific = {}  # contain task specific options
        for k, v in kwargs.items():
            key_prefix, key_sep, key_suffix = k.partition(self._DELIMITER)
            if key_prefix == self._PREFIX and key_sep == self._DELIMITER:
                self._specific[key_suffix] = v
            else:
                self._broadcast[k] = v

    def task_options(self, task_name: str) -> dict:
        """Return an option dict for a specific task following these rules.

        The original options include broadcast options and task-specific options.
        1. If the task_name is a not in the specific options, return a
        broadcast options dict.
        2. Otherwise, return a dict with broadcast options but overridden
        by the task-specific options.

        Args:
            task_name: The name of the task.

        Returns:
            dict: The options for the task.
        """
        res = copy.deepcopy(self._broadcast)
        if task_name not in self._specific:
            return res
        res.update(self._specific[task_name])
        return res
