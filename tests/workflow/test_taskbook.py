import textwrap

import pytest

from laboneq_applications.workflow import task
from laboneq_applications.workflow.taskbook import Task, TaskBook, taskbook


@task
def task_a():
    return 1


@task
def task_b():
    task_a()
    return 2


class TestTaskbook:
    def test_init(self):
        book = TaskBook()
        assert book.output is None
        assert book.tasks == []

    def test_add_entry(self):
        book = TaskBook()
        entry_a = Task(task=task_a, output=1, args=(), kwargs={})
        entry_b = Task(task=task_a, output=5, args=(), kwargs={})
        book.add_entry(entry_a)
        book.add_entry(entry_b)
        assert book.tasks == [entry_a, entry_b]

    def test_repr(self):
        book = TaskBook()
        entry_a = Task(task=task_a, output=1, args=(), kwargs={})
        entry_b = Task(task=task_a, output=5, args=(), kwargs={})
        book.add_entry(entry_a)
        book.add_entry(entry_b)
        assert str(book) == "TaskBook(tasks=[Task(name=task_a), Task(name=task_a)])"


class TestTask:
    def test_name(self):
        t = Task(task_a, 2, 3, 4)
        assert t.name == "task_a"

    def test_func(self):
        t = Task(task_a, 2, 3, 4)
        assert t.func == task_a.func

    def test_eq(self):
        e1 = Task(task_a, 2, 3, 4)
        e2 = Task(task_a, 2, 3, 4)
        assert e1 == e2

        e1 = Task(task_a, 2, 3, 4)
        e2 = Task(task_b, 2, 3, 4)
        assert e1 != e2

        e1 = Task(task_a, 2, 3, 4)
        assert e1 != 2
        assert e1 != "bar"

    def test_repr(self):
        t = Task(task_a, 2, 3, 4)
        assert str(t) == "Task(name=task_a)"

    def test_src(self):
        @task
        def task_():
            return 1

        t = Task(task_, 2, 3, 4)
        assert t.src == textwrap.dedent("""\
            @task
            def task_():
                return 1
        """)


class TestTaskBookDecorator:
    def test_result(self):
        @taskbook
        def book():
            return 123

        res = book()
        assert res.output == 123

    def test_task_called_multiple_times(self):
        @taskbook
        def book():
            task_a()
            task_a()

        result = book()
        assert result.tasks == [
            Task(task=task_a, output=1, args=(), kwargs={}),
            Task(task=task_a, output=1, args=(), kwargs={}),
        ]

    def test_nested_task_recorded(self):
        @taskbook
        def book():
            task_b()

        result = book()
        assert result.tasks == [
            Task(task=task_a, output=1, args=(), kwargs={}),
            Task(task=task_b, output=2, args=(), kwargs={}),
        ]

    def test_nested_taskbooks(self):
        @taskbook
        def book_a():
            task_b()

        @taskbook
        def book_b():
            book_a()

        with pytest.raises(NotImplementedError, match="Taskbooks cannot be nested."):
            book_b()

    def test_task_return_value_is_returned(self):
        @taskbook
        def book():
            return task_a()

        res = book()
        assert res.output == 1

    def test_run_normal_function_as_task(self):
        def normal_function(x):
            return x + 2

        @taskbook
        def book():
            task(normal_function)(1)
            return normal_function(3)

        res = book()
        assert len(res.tasks) == 1
        assert res.tasks[0].output == 3
        assert res.output == 5

    def test_normal_function_with_task(self):
        @task
        def add_1(x):
            return x + 1

        def sum_2(x):
            return x + 2

        @taskbook
        def book(x, y=None):
            for value in [1, x, y]:
                latest_record = add_1(value)
            return sum_2(latest_record)

        res = book(5, y=10)
        assert [r.output for r in res.tasks] == [2, 6, 11]
        assert res.output == 10 + 1 + 2
