"""This module defines objects related to taskbook.

A taskbook is a collection of tasks and their records.

Whenever a task is executed within a taskbook, its'
arguments, result and other relevant information is
saved into the taskbook records.

## Example

### Building and running the taskbook

```python
from laboneq_applications.workflow import task, taskbook

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

import inspect
import textwrap
import threading
from collections.abc import Sequence
from functools import update_wrapper
from typing import (
    TYPE_CHECKING,
    Callable,
    ClassVar,
    Generic,
    TypeVar,
    cast,
    overload,
)

from typing_extensions import ParamSpec

from laboneq_applications.core.options import TaskBookOptions
from laboneq_applications.workflow import _utils
from laboneq_applications.workflow._context import (
    ExecutorContext,
    LocalContext,
    get_active_context,
)
from laboneq_applications.workflow.exceptions import WorkflowError

if TYPE_CHECKING:
    from typing import Callable

    from laboneq_applications.workflow.task import task_


class _TaskBookStopExecution(BaseException):
    """Raised while a task book is running to end the execution."""


class Task:
    """A task.

    The instance holds execution information of an task when it
    was executed in a taskbook.

    Attributes:
        name: The name of the task.
        func: Underlying Python function.
        output: The output of the function.
        parameters: Input parameters of the task.
    """

    def __init__(
        self,
        task: task_,
        output: object,
        parameters: dict | None = None,
    ) -> None:
        self._task = task
        self._output = output
        self._parameters = parameters or {}
        self._taskbook: TaskBook | None = None

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

    @property
    def parameters(self) -> dict:
        """Input parameters of the task."""
        return self._parameters

    def rerun(
        self,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Rerun the task.

        Rerunning the task appends the results into the attached taskbook,
        and returns the return value of the task.

        The given parameters overwrite the original run parameters of the task,
        which can be inspected with `Task.parameters`. Therefore it is possible
        to supply only specific parameters that one wishes to change and rerun
        the task.
        It is recommended that in the case of an partial parameters, keyword
        arguments are used.

        Arguments:
            *args: Arguments forwarded into the task.
            **kwargs: Keyword arguments forwarded into the original task.

        Returns:
            The return value of the task.
        """
        # TODO: Fill args, kwargs with previous and overwrite with given arguments.
        kwargs = kwargs if kwargs is not None else {}
        args = args if args is not None else ()
        sig = inspect.signature(self.func)
        args_partial = sig.bind_partial(*args, **kwargs)
        params = self.parameters | args_partial.arguments
        r = self.func(**params)
        if self._taskbook is not None:
            entry = Task(
                task=self._task,
                output=r,
                parameters=params,
            )
            self._taskbook.add_entry(entry)
        return r

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Task):
            return NotImplemented
        return (
            self._task == value._task
            and self.output == value.output
            and self.parameters == value.parameters
        )

    def __repr__(self) -> str:
        attrs = ", ".join(
            [
                f"name={self.name}",
                f"output={self.output}",
                f"parameters={self.parameters}",
                f"func={self.func}",
            ],
        )
        return f"Task({attrs})"

    def __str__(self) -> str:
        return f"Task({self.name})"

    def _repr_pretty_(self, p, cycle):  # noqa: ANN001, ANN202, ARG002
        # For Notebooks
        p.text(str(self))


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
        return repr(self._tasks)

    def __str__(self) -> str:
        return ", ".join([str(t) for t in self._tasks])

    def _repr_pretty_(self, p, cycle):  # noqa: ANN001, ANN202, ARG002
        # For Notebooks
        p.text(str(self))

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


ReturnType = TypeVar("ReturnType")


class TaskBook(Generic[ReturnType]):
    """A taskbook.

    A taskbook is a collection of executed tasks and their results.
    """

    def __init__(self, parameters: dict | None = None) -> None:
        self._parameters = parameters or {}
        self._tasks: list[Task] = []
        self._output: ReturnType | None = None

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
    def parameters(self) -> dict:
        """Input parameters of the taskbook."""
        return self._parameters

    @property
    def output(self) -> ReturnType:
        """Output of the taskbook."""
        return cast(ReturnType, self._output)

    def add_entry(self, entry: Task) -> None:
        """Add an entry to the taskbook.

        Arguments:
            entry: Task entry.
        """
        if entry._taskbook is not None:
            raise WorkflowError("Task is already attached to an taskbook.")
        entry._taskbook = self
        self._tasks.append(entry)

    def __repr__(self) -> str:
        attrs = ", ".join(
            [
                f"output={self.output}",
                f"parameters={self.parameters}",
                f"tasks={self.tasks!r}",
            ],
        )
        return f"TaskBook({attrs})"

    def __str__(self) -> str:
        return f"Taskbook\nTasks: {self.tasks}"

    def _repr_pretty_(self, p, cycle):  # noqa: ANN001, ANN202, ARG002
        # For Notebooks
        p.text(str(self))


class _TaskBookExecutor(ExecutorContext):
    """A taskbook executor."""

    def __init__(
        self,
        taskbook: TaskBook,
        options: TaskBookOptions,
    ) -> None:
        self.taskbook = taskbook
        ctx = get_active_context()
        if isinstance(ctx, _TaskBookExecutor):
            # TODO: Should nested books append to the top level or?
            raise NotImplementedError("Taskbooks cannot be nested.")
        self.options = options
        self._run_until = self.options.run_until

    def __enter__(self):
        LocalContext.enter(self)

    def __exit__(self, exc_type, exc_value, traceback):  # noqa: ANN001
        LocalContext.exit()
        return isinstance(exc_value, _TaskBookStopExecution)

    def execute_task(
        self,
        task: task_,
        *args: object,
        **kwargs: object,
    ) -> None:
        # TODO: Error handling and saving of the exception during execution
        if hasattr(self.options, task.name):
            if task.has_opts:
                kwargs["options"] = getattr(self.options, task.name)
            else:
                raise ValueError(f"Task {task.name} does not require options.")

        r = task._run(*args, **kwargs)
        entry = Task(
            task=task,
            output=r,
            parameters=_utils.create_argument_map(task.func, *args, **kwargs),
        )
        self.taskbook.add_entry(entry)
        if self._run_until is not None and task.name == self._run_until:
            raise _TaskBookStopExecution(task.name)
        return r


class _ContextStorage(threading.local):
    # NOTE: Subclassed for type hinting
    results: ClassVar[dict[str, TaskBook]] = {}


_results = _ContextStorage()

Parameters = ParamSpec("Parameters")


class taskbook_(Generic[Parameters, ReturnType]):  # noqa: N801
    """Task book wrapper for a function."""

    def __init__(
        self,
        func: Callable[Parameters, ReturnType],
        options: type[TaskBookOptions] | None = None,
    ) -> None:
        self._func = func
        self.__doc__ = self._func.__doc__
        if options is None:
            self._options = None
        elif isinstance(options, type) and issubclass(options, TaskBookOptions):
            self._options = options
        else:
            raise TypeError(
                "Options must be a subclass of TaskBookOptions.",
            )

    @property
    def func(self) -> Callable[Parameters, ReturnType]:
        """The underlying function."""
        return self._func

    @property
    def options(self) -> type[TaskBookOptions]:
        """The options for the taskbook.

        Raise:
            AttributeError: If the taskbook does not have options declared.
        """
        if self._options is None:
            raise AttributeError("Taskbook does not have options declared.")
        return self._options

    @property
    def src(self) -> str:
        """Source code of the underlying function."""
        src = inspect.getsource(self._func)
        return textwrap.dedent(src)

    def _func_full_path(self) -> str:
        return ".".join([self.func.__module__, self.func.__qualname__])

    def __call__(  # noqa: D102
        self,
        *args: Parameters.args,
        **kwargs: Parameters.kwargs,
    ) -> TaskBook[ReturnType]:
        if self._options is not None:
            opts_input = kwargs.get("options", self._options())
            if isinstance(opts_input, dict):
                opt = self._options(**opts_input)
            elif isinstance(opts_input, self._options):
                opt = opts_input
            else:
                raise TypeError(
                    "Options must be a dictionary or an instance of TaskBookOptions.",
                )
        else:
            opt = TaskBookOptions()
        book = TaskBook(
            parameters=_utils.create_argument_map(self.func, *args, **kwargs),
        )

        with _TaskBookExecutor(
            book,
            options=opt,
        ):
            try:
                book._output = self._func(*args, **kwargs)
            except Exception:
                _results.results[self._func_full_path()] = book
                raise
        return book

    def recover(self) -> TaskBook[ReturnType]:
        """Recover the taskbook of the latest run that raised an exception.

        The value will be populated only if an exception is raised
        from the taskbook function.
        Getting the latest value will pop it from the memory and only one
        result is stored per taskbook.

        Returns:
            Latest taskbook that raised an exception.

        Raises:
            WorkflowError: Taskbook has no previous record.
        """
        try:
            return _results.results.pop(self._func_full_path())
        except KeyError as error:
            raise WorkflowError("Taskbook has no previous record.") from error


def taskbook(
    func: Callable[Parameters, ReturnType] | None = None,
    options: type[TaskBookOptions] | None = None,
) -> taskbook_[Parameters, ReturnType]:
    """A decorator to turn a function into a taskbook.

    When a function is marked as a taskbook, it will record
    each task's information.

    ### Storing the results in case of an error

    When a function wrapped as a taskbook raises an Exception,
    the partial result up until that point can be retrieved by using
    `.recover()` method.

    Arguments:
        func: Function to be marked as a taskbook.
        options: Options for the taskbook.
    """
    if func is None:

        def decorator(func):  # noqa: ANN202, ANN001
            out = update_wrapper(
                taskbook_(func, options=options),
                func,
                assigned=(
                    "__module__",
                    "__name__",
                    "__qualname__",
                    "__annotations__",
                    "__type_params__",
                    "__doc__",
                ),
            )
            return cast(taskbook_[Parameters, ReturnType], out)

        # TODO: Fix the type of the decorator
        return decorator
    out = update_wrapper(
        taskbook_(func),
        func,
        assigned=(
            "__module__",
            "__name__",
            "__qualname__",
            "__annotations__",
            "__type_params__",
            "__doc__",
        ),
    )
    return cast(taskbook_[Parameters, ReturnType], out)
