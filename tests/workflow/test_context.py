import pytest

from laboneq_applications.workflow._context import LocalContext


def test_no_local_context():
    with pytest.raises(RuntimeError):
        LocalContext().exit()

    with pytest.raises(RuntimeError):
        LocalContext().active_context()
