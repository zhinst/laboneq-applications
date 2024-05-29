import textwrap

from laboneq_applications.workflow.engine.task_block import TaskBlock
from laboneq_applications.workflow.task import task


@task
def addition(): ...


class TestTaskBlock:
    def test_name(self):
        blk = TaskBlock(addition)
        assert blk.name == "addition"

    def test_repr(self):
        blk = TaskBlock(addition)
        assert str(blk) == "Task(name=addition)"

    def test_execute(self):
        @task
        def addition(x, y):
            return x + y

        blk = TaskBlock(addition, 1, y=2)
        r = blk.execute()
        assert r.log == {"addition": [3]}

    def test_src(self):
        @task
        def addition(x, y):
            return x + y

        blk = TaskBlock(addition, 1, y=2)
        assert blk.src == textwrap.dedent("""\
            @task
            def addition(x, y):
                return x + y
        """)
