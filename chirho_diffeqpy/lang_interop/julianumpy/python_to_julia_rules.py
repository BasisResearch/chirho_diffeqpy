import numpy as np

from ..ops import convert_python_to_julia
from .internals import JuliaThingWrapper, _JuliaThingWrapperArray


@convert_python_to_julia.register
def _(thing: object, out=None, out_dtype=None):
    # A new default implementation — instead of throwing an error just try to wrap whatever didn't hit the types below.
    # This implementation will kick in if either the type isn't wrapped properly, and is some julia type,
    #  or is a native python type because the function return does not depend on the argument types.
    # This can happen, e.g. if a function with default native python arguments is called with no arguments.
    return thing


@convert_python_to_julia.register
def _(thing: JuliaThingWrapper, out=None, out_dtype=None):
    return thing.julia_thing


@convert_python_to_julia.register
def _(thing: _JuliaThingWrapperArray, out=None, out_dtype=None):
    return JuliaThingWrapper.unwrap_array(thing, out=out, out_dtype=out_dtype)


@convert_python_to_julia.register
def _(thing: np.ndarray, out=None, out_dtype=None):
    # This handles unwrapped numpy arrays of wrapped julia types. Recast as view through _JuliathingWrapperArray,
    #  and then convert as specified elsewhere. Note that we don't rewrap, as that defaults to also wrapping
    #  all the elements in a JuliaThingWrapper.
    return convert_python_to_julia(
        thing.view(_JuliaThingWrapperArray), out=out, out_dtype=out_dtype
    )


@convert_python_to_julia.register(tuple)  # to support multiple return values.
def _(thing, out=None, out_dtype=None):
    return tuple(
        convert_python_to_julia(inner_thing, out=out, out_dtype=out_dtype)
        for inner_thing in thing
    )
