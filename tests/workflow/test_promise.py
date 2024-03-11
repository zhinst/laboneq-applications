import pytest

from laboneq_library.workflow.promise import Promise


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

    def test_head_not_resolved_error(self):
        head = Promise()
        child = head["key"]
        with pytest.raises(RuntimeError):
            child.result()

    def test_child_cannot_be_resolved(self):
        head = Promise()
        child = head["key"]
        with pytest.raises(ValueError):
            child.set_result({"key": 0})
