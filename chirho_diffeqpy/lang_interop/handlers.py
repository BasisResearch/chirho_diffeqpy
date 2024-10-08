from functools import wraps

from .ops import convert_julia_to_python, convert_python_to_julia


def callable_from_julia(out_as_first_arg: bool = False, **python_to_julia_kwargs):
    """
    A backend agnostic decorator that supports wrapping julia inputs to a python function,
    and then unwrapping the python output to something compatible with julia.

    :param out_as_first_arg: If True, the first argument of the function will be treated as the function output,
        which will be modified in place.
    :param python_to_julia_kwargs: Additional keyword arguments to pass to `convert_python_to_julia`. No arguments
        can be passed to convert_julia_to_python.
    """

    def parameterized_decorator_closure(f):

        def julia_type_converter_wrapper(*args, out=None, **kwargs):

            result = f(
                *convert_julia_to_python(args), **convert_julia_to_python(kwargs)
            )

            converted_result = convert_python_to_julia(
                result, out=out, **python_to_julia_kwargs
            )

            if out is not None:
                return None
            else:
                return converted_result

        if out_as_first_arg:
            return wraps(f)(
                lambda out, *args, **kwargs: julia_type_converter_wrapper(
                    *args, out=out, **kwargs
                )
            )
        else:
            return wraps(f)(julia_type_converter_wrapper)

    return parameterized_decorator_closure
