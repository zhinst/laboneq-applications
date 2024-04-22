from __future__ import annotations

import threading
from typing import Any, ClassVar


class InstanceRegister:
    """A register to keep track of instances within the context."""

    def __init__(self):
        self.instances: list[Any] = []

    def register(self, instance: Any) -> None:  # noqa: ANN401
        self.instances.append(instance)


class _ContextStorage(threading.local):
    # NOTE: Subclassed for type hinting
    active_contexts: ClassVar[list[LocalContext]] = []


class LocalContext:
    _contexts = _ContextStorage()

    def __init__(self):
        self.ctx = InstanceRegister()

    @classmethod
    def enter(cls) -> None:
        cls = LocalContext()
        LocalContext._contexts.active_contexts.append(cls)

    @classmethod
    def exit(cls) -> InstanceRegister:
        if LocalContext._contexts.active_contexts:
            return LocalContext._contexts.active_contexts.pop().ctx
        msg = "No active Workflow context"
        raise RuntimeError(msg)

    @classmethod
    def is_active(cls) -> bool:
        return len(LocalContext._contexts.active_contexts) != 0

    @classmethod
    def active_context(cls) -> InstanceRegister:
        if LocalContext._contexts.active_contexts:
            return LocalContext._contexts.active_contexts[-1].ctx
        msg = "No active Workflow context"
        raise RuntimeError(msg)
