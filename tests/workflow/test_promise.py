import pytest

from laboneq_applications.workflow.promise import Promise


class TestPromise:
    def test_head_getitem(self):
        head = Promise()
        child = head["key"]
        head.set_result({"key": 123})
        assert child.result() == 123

        head = Promise()
        child = head[0]["key"]
        head.set_result(({"key": 123},))
        assert child.result() == 123

    def test_promise_done_state(self):
        head = Promise()
        child = head["key"]
        assert head.done is False
        assert child.done is False
        head.set_result(1)
        assert head.done is True
        assert child.done is True

    def test_head_not_resolved_error(self):
        head = Promise()
        child = head["key"]
        with pytest.raises(RuntimeError, match="Promise not resolved"):
            child.result()

    def test_child_cannot_be_resolved(self):
        head = Promise()
        child = head["key"]
        with pytest.raises(ValueError, match="Cannot resolve child promises"):
            child.set_result({"key": 0})
