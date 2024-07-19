# Adapted from https://stackoverflow.com/a/36837332
from typing import Dict, Tuple, Callable, Optional
from functools import wraps

RegistryType = Dict[object, Callable]


# Make a publically accessible valuedispatch decorator that returns a _ValueDispatcher with the original
#  function as its default.
def singledispatch_on_value_and_test_arg(default_func: Callable):
    """
    Decorates a function that allows for value dispatch based on how that value hashes (values that hash the same
     as a value with a registered function will be dispatched to that function).
    :param default_func:
    :return:
    """
    return _ValueSingleDispatcher(default_func)


valuedispatch = singledispatch_on_value_and_test_arg


class _ValueSingleDispatcher:

    def __init__(self, default_func: Callable):
        # Registry that is used for value dispatch based on a specific test, arg pair.
        self.test_arg_specific_registry: Dict[Tuple[str, str], RegistryType] = dict()

        # Registry that is used for general value dispatch in the event that a test-specific dispatch
        #  is not defined.
        self.general_registry: RegistryType = dict()

        self.default_func = default_func

    @staticmethod
    def _build_test_arg_key(testid: Optional[str], argnames: Optional[str]) -> Optional[Tuple[str, str]]:
        # Require that both or neither of the testid or argnames args are provided.
        if (testid is None) != (argnames is None):
            raise ValueError("Both or neither of testid and argnames must be provided.")

        if testid is None:
            return None
        else:
            return testid, argnames

    def _retrieve_registry(self, testid: Optional[str], argnames: Optional[str]):
        test_arg_key = self._build_test_arg_key(testid, argnames)
        return self.test_arg_specific_registry.get(test_arg_key, self.general_registry)

    def __call__(self, value: object, *args, testid: Optional[str] = None, argnames: Optional[str] = None, **kwargs):
        func = self._retrieve_registry(testid, argnames).get(value, None)

        if func is None:
            self.default_func(value, *args, **kwargs)
        else:
            return func(value, *args, **kwargs)

    def register(self, value: object, testid: Optional[str] = None, argnames: Optional[str] = None):

        test_arg_key = self._build_test_arg_key(testid, argnames)

        if test_arg_key is not None:
            if test_arg_key not in self.test_arg_specific_registry:
                self.test_arg_specific_registry[test_arg_key] = dict()

        registry = self._retrieve_registry(testid, argnames)

        def decorator(func: Callable):
            registry[value] = func
            return func

        return decorator
