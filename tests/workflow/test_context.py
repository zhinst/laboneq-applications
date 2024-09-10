import pytest

from laboneq_applications.workflow._context import (
    LocalContext,
)


class TestLocalContext:
    def test_no_local_context(self):
        with pytest.raises(RuntimeError, match="No active context."):
            assert LocalContext().exit()
        assert LocalContext.get_active() is None

    def test_exit_in_scope(self):
        with pytest.raises(RuntimeError, match="No active context."):
            with LocalContext.scoped():
                LocalContext().exit()

        with pytest.raises(RuntimeError, match="No active context."):  # noqa: PT012
            with LocalContext.scoped():
                with LocalContext.scoped():
                    LocalContext().exit()

    def test_active_context(self):
        assert LocalContext.get_active() is None
        with LocalContext.scoped():
            assert LocalContext.get_active() is None
        assert LocalContext.get_active() is None

    def test_handler(self):
        handler = {"a": 5}
        with LocalContext.scoped(handler):
            # Nested scope
            with LocalContext.scoped():
                assert LocalContext.get_active() is None
            assert LocalContext.get_active() == handler


def test_get_active_context():
    assert LocalContext.get_active() is None
    with LocalContext.scoped(1):
        assert LocalContext.get_active() == 1
    assert LocalContext.get_active() is None
