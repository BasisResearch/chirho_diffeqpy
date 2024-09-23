from functools import singledispatch
from typing import Callable, Dict, Optional, Tuple

from .singledispatch_on_value import singledispatch_on_value

_scopes: Dict[str, Tuple[Callable, Callable]] = dict()


# A helper function to create a pair of value and type dispatchable functions.
def _create_get__dispatch_scope(scope: str):

    if scope in _scopes:
        return _scopes[scope]

    @singledispatch
    def by_type(dispatch_arg, *args, **kwargs):
        raise NotImplementedError

    # 1. Reparametrization starts with an attempt at value dispatch, which then falls back to a general type dispatch.
    @singledispatch_on_value
    def by_value(dispatch_arg, *args, **kwargs):
        # 2. And if no specific dispatcher is found for the specific value, we fall back to the default implementation
        #  here, which tries type dispatch.
        # return _scopes[scope][-1](dispatch_arg, *args, **kwargs)
        return by_type(dispatch_arg, *args, **kwargs)

    _scopes[scope] = by_value, by_type
    return by_value, by_type


# Global, fallback scope.
_global_by_value, _ = _create_get__dispatch_scope("global")


class _ReparametrizeWithinScope:
    """
    Reparametrize according to a particular scope, falling back to the global scope if the dispatch fails and
    the scope wasn't already global.
    """

    def __call__(self, dispatch_arg, *args, scope: str = "global", **kwargs):
        fail_msg = (
            f"Failed to find dispatch by value or by type, at first for scope {scope},"
            f" then globally, for {dispatch_arg} of type {type(dispatch_arg)}"
        )

        # Try in this scope.
        by_value, _ = _create_get__dispatch_scope(scope)
        try:
            return by_value(dispatch_arg, *args, **kwargs)
        except NotImplementedError:
            # If we just failed at the global level, raise here.
            if scope == "global":
                raise NotImplementedError(fail_msg)

            # Otherwise, try again explicitly at the global level.
            try:
                return _global_by_value(dispatch_arg, *args, **kwargs)
            except NotImplementedError:
                raise NotImplementedError(fail_msg)


class _ReparametrizeArgument(_ReparametrizeWithinScope):
    """
    A thin wrapper that is provides a register decorator akin to singledispatch's register, but for value dispatch.
    Also is callable, see _ReparametrizeByScope.
    """

    @staticmethod
    def register(dispatch_arg, scope: str = "global"):

        by_value, _ = _create_get__dispatch_scope(scope)

        def decorator(func):
            by_value.register(dispatch_arg)(func)
            return func

        return decorator


class _ReparametrizeArgumentByType(_ReparametrizeWithinScope):
    """
    A thin wrapper that provides a register decorator matching the singledispatch register decorator.
    """

    @staticmethod
    def register(dispatch_type: Optional = None, scope: str = "global"):

        _, by_type = _create_get__dispatch_scope(scope)

        def decorator(func):
            if dispatch_type is not None:
                by_type.register(dispatch_type)(func)
            else:
                by_type.register()(func)
            return func

        return decorator
