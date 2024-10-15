import pytest

from laboneq_applications.workflow import _utils


class TestCreateArguments:
    def test_overwrite_default(self):
        def func(x, y, b=2, c=True): ...  # noqa: FBT002

        r = _utils.create_argument_map(func, 1, 2, b=5, c=False)
        assert r == {"x": 1, "y": 2, "b": 5, "c": False}

    def test_default_applied(self):
        def func(x, y, b=2, c=True): ...  # noqa: FBT002

        r = _utils.create_argument_map(func, 1, 2)
        assert r == {"x": 1, "y": 2, "b": 2, "c": True}

    def test_missing_arguments(self):
        def func(x, y, b=2, c=True): ...  # noqa: FBT002

        with pytest.raises(TypeError):
            _utils.create_argument_map(func, 1)

        with pytest.raises(TypeError):
            _utils.create_argument_map(func, 1, 2, bar="Test")

    def test_kwargs(self):
        def func(x, **kwargs): ...

        r = _utils.create_argument_map(func, 1, y=2)
        assert r == {"x": 1, "y": 2}
