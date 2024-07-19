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
        try:
            func = self.registry.get(value, None)
        except TypeError as e:
            # See error reporting in the register method below.
            if "unhashable type" in str(e):
                func = None
            else:
                raise

        if func is None:
            return self.default_func(value, *args, **kwargs)
        else:
            return func(value, *args, **kwargs)

    def register(self, value: object):
        def decorator(func: Callable):
            try:
                self.registry[value] = func
            except TypeError as e:
                if "unhashable type" in str(e):
                    raise TypeError(f"Value {value} is not hashable, and cannot be used for singledispatch_on_value."
                                    f" You can either use type dispatch, or use type dispatch on the unhashable form "
                                    f" and convert to a hashable form, then dispatch by value on the hashable form.")
            return func

        return decorator
