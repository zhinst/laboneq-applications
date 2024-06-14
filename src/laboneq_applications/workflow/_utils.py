import inspect
from typing import Callable


def create_argument_map(func: Callable, *args: object, **kwargs: object) -> dict:
    """Create a mapping out of function arguments.

    Arguments:
        func: Callable
        *args: Arguments of the callable.
        **kwargs: Keyword arguments of the callable.

    Returns:
        An ordered dict of the arguments.
    """
    sig = inspect.signature(func)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)
