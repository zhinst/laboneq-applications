"""Module for argument resolvers."""

from __future__ import annotations

from laboneq_applications.workflow.engine.promise import Promise


class ArgumentResolver:
    """Argument resolver.

    Resolves promise arguments into requirements, which
    includes promises that are required to run before the arguments
    can be fully resolves.
    """

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._promises_args: list[tuple[int, Promise]] = []
        self._promises_kwargs: dict[str, Promise] = {}

        for idx, arg in enumerate(self._args):
            if isinstance(arg, Promise):
                self._promises_args.append((idx, arg))
        for key, arg in self._kwargs.items():
            if isinstance(arg, Promise):
                self._promises_kwargs[key] = arg

    @property
    def args(self) -> tuple:
        """Arguments to resolve."""
        return self._args

    @property
    def kwargs(self) -> dict:
        """Keyword arguments to resolve."""
        return self._kwargs

    def resolve(self) -> tuple[tuple, dict]:
        """Resolve input arguments.

        Maps input to the corresponding arguments used.
        """
        args = list(self.args)
        kwargs = self.kwargs
        for idx, arg in self._promises_args:
            args[idx] = arg.result()
        for key, kwarg in self._promises_kwargs.items():
            kwargs[key] = kwarg.result()
        return tuple(args), kwargs
