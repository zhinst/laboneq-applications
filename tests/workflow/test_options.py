from __future__ import annotations

import pytest
from pydantic import ValidationError

from laboneq_applications.workflow.options import (
    WorkflowOptions,
)


class A(WorkflowOptions): ...


class TestWorkflowOptions:
    def test_create_opt(self):
        class A(WorkflowOptions):
            alice: int = 1

        a = A()
        assert a.alice == 1

        # option attributes can be updated
        a.alice = 2
        assert a.alice == 2

    def test_create_nested(self):
        class Nested(WorkflowOptions):
            bob: int = 1

        class A(WorkflowOptions):
            alice: int = 1
            nested: Nested = Nested()

        a = A()
        assert a.alice == 1
        assert isinstance(a.nested, Nested)
        assert a.nested.bob == 1
        # option attributes can be updated
        a.alice = 2
        a.nested.bob = 2
        assert a.alice == 2
        assert a.nested.bob == 2

    def test_validate_options(self):
        # Initialize option with non-existing attributes will raises error
        with pytest.raises(ValidationError):
            WorkflowOptions(non_existing=1)
