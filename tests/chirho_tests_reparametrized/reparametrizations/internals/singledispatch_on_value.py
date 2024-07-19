from typing import Callable, Optional, Dict


# Make a publically accessible valuedispatch decorator that returns a _ValueDispatcher with the original
#  function as its default.
def singledispatch_on_value(default_func: Callable):
    """
    Decorates a function that allows for value dispatch based on how that value hashes (values that hash the same
     as a value with a registered function will be dispatched to that function).
    This is primarily used for defining dispatch on a particular test fixture.
    :param default_func:
    :return:
    """
    return _SingleDispatcherOnValue(default_func)


class _SingleDispatcherOnValue:

    def __init__(self, default_func: Callable):
        self.registry: Dict[object, Callable] = dict()
        self.default_func = default_func

    def __call__(self, value: object, *args, testid: Optional[str] = None, argnames: Optional[str] = None, **kwargs):
        func = self.registry.get(value, None)

        if func is None:
            self.default_func(value, *args, **kwargs)
        else:
            return func(value, *args, **kwargs)

    def register(self, value: object):
        def decorator(func: Callable):
            self.registry[value] = func
            return func

        return decorator
