from functools import singledispatch


@singledispatch
def convert_julia_to_python(v, *args, **kwargs):
    raise NotImplementedError(
        f"Conversion from {type(v)} to Python not implemented. Did you forget to import"
        f" the appropriate backend implementations?"
    )


@singledispatch
def convert_python_to_julia(v, *args, out=None, **kwargs):
    raise NotImplementedError(
        f"Conversion of {type(v)} to Julia not implemented. Did you forget to import"
        f" the appropriate backend implementations?"
    )
