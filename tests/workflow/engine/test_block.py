import textwrap

import pytest

from laboneq_applications.workflow.engine.block import Block, BlockResult, TaskBlock
from laboneq_applications.workflow.engine.promise import (
    Promise,
    PromiseResultNotResolvedError,
    ReferencePromise,
)
from laboneq_applications.workflow.task import task


class TestBlockResult:
    def test_log(self):
        res = BlockResult()
        assert res.log == {}

    def test_add_result(self):
        res = BlockResult()
        res.add_result("a", [1, 2])
        res.add_result("b", 3)
        assert res.log == {"a": [[1, 2]], "b": [3]}

    def test_merge(self):
        res_a = BlockResult()
        res_a.add_result("a", [1, 2])
        res_a.add_result("b", 3)
        res_b = BlockResult()
        res_b.add_result("a", [2])
        res_a.merge(res_b)
        assert res_a.log == {"a": [[1, 2], [2]], "b": [3]}
        assert res_b.log == {"a": [[2]]}


class TBlock(Block):
    def execute(self) -> BlockResult:
        result = BlockResult()
        for blk in self.body:
            result.merge(self._run_block(blk))
        return result


class TestBlock:
    def test_name(self):
        a = TBlock()
        assert a.name == "TBlock"

    def test_arguments(self):
        a = TBlock(1, 2, b="bar")
        args, kwargs = a._resolver.resolve()
        assert args == (1, 2)
        assert kwargs == {"b": "bar"}

    def test_context(self):
        class MinusBlock(Block):
            def execute(self) -> BlockResult:
                r = BlockResult()
                r.add_result("res", 1)
                return r

        a = TBlock()
        b = MinusBlock(1, 2, a="foo", b="bar")
        with a:
            with b:
                ...
        assert a.body == [b]
        res = a.execute()
        assert res.log == {"res": [1]}


class ResolverBlock(Block):
    def execute(self) -> BlockResult:
        self._resolver.resolve()


class TestBlockPromiseArgumentNotResolved:
    def test_promise(self):
        promise = Promise()
        blk = ResolverBlock(promise)
        with pytest.raises(
            PromiseResultNotResolvedError,
            match="Promise result is not resolved.",
        ):
            blk.execute()

    def test_reference_promise(self):
        class SomeReferenceObject:
            def __repr__(self) -> str:
                return "SomeObject"

        o = SomeReferenceObject()
        promise = ReferencePromise(o)
        blk = ResolverBlock(promise)
        with pytest.raises(
            PromiseResultNotResolvedError,
            match="Result for 'SomeObject' is not resolved.",
        ):
            blk.execute()


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
