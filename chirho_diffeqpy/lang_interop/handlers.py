from functools import wraps
from .ops import convert_julia_to_python, convert_python_to_julia


def callable_from_julia(f, **python_to_julia_kwargs):
    """
    TODO
    note that out is reserved for conversions making use of in-place operations.
    """

    @wraps(f)
    def wrapper(*args, out=None, **kwargs):

        print("about to convert args and kwargs")

        result = f(
            *convert_julia_to_python(args),
            **convert_julia_to_python(kwargs)
        )

        converted_result = convert_python_to_julia(result, out=out, **python_to_julia_kwargs)
        print("-----------------")
        print("type(converted_result):", type(converted_result))
        print("converted_result:", converted_result)
        return converted_result

    return wrapper
