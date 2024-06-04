import pytest

from laboneq_applications.workflow import task
from laboneq_applications.workflow.taskbook import TaskBook, TaskEntry, taskbook


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
        assert book.result is None
        assert book.tasks == []

    def test_add_entry(self):
        book = TaskBook()
        entry_a = TaskEntry(task=task_a, result=1, args=(), kwargs={})
        entry_b = TaskEntry(task=task_a, result=5, args=(), kwargs={})
        book.add_entry(entry_a)
        book.add_entry(entry_b)
        assert book.tasks == [entry_a, entry_b]


class TestTaskEntry:
    def test_eq(self):
        e1 = TaskEntry(task_a, 2, 3, 4)
        e2 = TaskEntry(task_a, 2, 3, 4)
        assert e1 == e2

        e1 = TaskEntry(task_a, 2, 3, 4)
        e2 = TaskEntry(task_b, 2, 3, 4)
        assert e1 != e2

        e1 = TaskEntry(task_a, 2, 3, 4)
        assert e1 != 2
        assert e1 != "bar"


class TestTaskBookDecorator:
    def test_result(self):
        @taskbook
        def book():
            return 123

        res = book()
        assert res.result == 123

    def test_task_called_multiple_times(self):
        @taskbook
        def book():
            task_a()
            task_a()

        result = book()
        assert result.tasks == [
            TaskEntry(task=task_a, result=1, args=(), kwargs={}),
            TaskEntry(task=task_a, result=1, args=(), kwargs={}),
        ]

    def test_nested_task_recorded(self):
        @taskbook
        def book():
            task_b()

        result = book()
        assert result.tasks == [
            TaskEntry(task=task_a, result=1, args=(), kwargs={}),
            TaskEntry(task=task_b, result=2, args=(), kwargs={}),
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
        assert res.result == 1

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
        assert [r.result for r in res.tasks] == [2, 6, 11]
        assert res.result == 10 + 1 + 2
