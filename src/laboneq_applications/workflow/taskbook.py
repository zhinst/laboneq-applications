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

The taskbook return value and the information of each task execution
can be accessed from the result object:

```python
>>> result.result
"success"
>>> result.tasks[0].result
7
>>> result.tasks[0].args
(6,)
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
    x = task(addition(1, x))  # Record will be saved
    y = addition(2, x)  # No record saved
```

Now the normal Python function `addition()` wrapped in `task()` will have records
within the taskbook.
"""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING

from laboneq_applications.workflow._context import (
    ExecutorContext,
    LocalContext,
    get_active_context,
)

if TYPE_CHECKING:
    from typing import Callable

    from laboneq_applications.workflow.task import Task


class TaskEntry:
    """A task entry.

    The instance holds execution information of an task when it
    was executed in a taskbook.

    Attributes:
        task: Task associated with the entry.
        result: The result produced by the task.
        args: Arguments of the task.
        kwargs: Keyword arguments of the task.
    """

    def __init__(
        self,
        task: Task,
        result: object,
        args: object,
        kwargs: object,
    ) -> None:
        self.task = task
        self.result = result
        self.args = args
        self.kwargs = kwargs

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, TaskEntry):
            return NotImplemented
        return (
            self.task == value.task
            and self.result == value.result
            and self.args == value.args
            and self.kwargs == value.kwargs
        )


class TaskBook:
    """A taskbook.

    A taskbook is a collection of executed tasks and their results.
    """

    def __init__(self) -> None:
        self._tasks: list[TaskEntry] = []
        self._result = None

    @property
    def tasks(self) -> list[TaskEntry]:
        """Task entries of the taskbook."""
        return self._tasks

    @property
    def result(self) -> object:
        """Result of the taskbook."""
        return self._result

    def add_entry(self, entry: TaskEntry) -> None:
        """Add an entry to the taskbook.

        Arguments:
            entry: Task entry.
        """
        self._tasks.append(entry)


class _TaskBookExecutor(ExecutorContext):
    """Taskbook executor."""

    def __init__(self, taskbook: TaskBook) -> None:
        self.taskbook = taskbook

    def execute_task(
        self,
        task: Task,
        *args: object,
        **kwargs: object,
    ) -> None:
        # TODO: Error handling and saving of the exception during execution
        r = task._run(*args, **kwargs)
        entry = TaskEntry(task=task, result=r, args=args, kwargs=kwargs)
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
        book = TaskBook()
        with LocalContext.scoped(_TaskBookExecutor(book)):
            book._result = func(*args, **kwargs)
        return book

    return inner
