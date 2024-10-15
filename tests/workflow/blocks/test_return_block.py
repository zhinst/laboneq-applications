import re

import pytest

from laboneq_applications.workflow import blocks


def test_return_outside_of_workflow_context():
    assert blocks.return_() is None
    assert blocks.return_(1) is None
    assert blocks.return_(x=3) is None


def test_return_invalid_arguments():
    msg = re.escape(
        "return_() takes either a single positional argument or keyword arguments"
    )
    with pytest.raises(TypeError, match=msg):
        blocks.return_(1, a=2)


class TestReturnExpression:
    def test_init(self):
        obj = blocks.ReturnStatement.from_value(1)
        assert obj.callback(value=5) == 5
        assert obj.parameters == {"value": 1}

        cb = lambda x: x  # noqa: E731
        obj = blocks.ReturnStatement(cb, x=1)
        assert obj.callback == cb
        assert obj.parameters == {"x": 1}


class TestNamespace:
    @pytest.fixture(scope="class")
    def namespace(self):
        return blocks.Namespace(x=1, y=2)

    def test_attribute_access(self, namespace):
        assert namespace.x == 1
        assert namespace.y == 2

    def test_dir(self, namespace):
        dir_ = dir(namespace)
        assert "x" in dir_
        assert "y" in dir_

    def test_vars(self, namespace):
        assert vars(namespace) == {"x": 1, "y": 2}

    def test_repr(self, namespace):
        assert repr(namespace) == "Namespace(x=1, y=2)"

    def test_invalid_attribute_access(self, namespace):
        with pytest.raises(
            AttributeError, match="'Namespace' object has no attribute 'foobar'"
        ):
            _ = namespace.foobar

    def test_immutability(self, namespace):
        with pytest.raises(AttributeError, match="can't set attribute"):
            namespace.x = 5

        with pytest.raises(AttributeError, match="can't delete attribute"):
            del namespace.x

        # Test modify __dict__, no restriction for now
        namespace.__dict__["x"] = 7
        assert namespace.x == 7
