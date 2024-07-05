from .internals import JuliaThingWrapper
from ..ops import convert_julia_to_python
import numpy as np


@convert_julia_to_python.register
def _(v: object):
    # A new default implementation — instead of throwing an error just try to wrap whatever didn't hit the types below.
    # This could be a scalar of many different julia types, so we aren't checking for those here. Note that this
    #  passive approach will break downstream somewhere if the wrapped julia thing isn't numerical in nature.
    return JuliaThingWrapper(v)


@convert_julia_to_python.register
def _(v: np.ndarray):
    return JuliaThingWrapper.wrap_array(v)


@convert_julia_to_python.register
def _(v: list):
    return [convert_julia_to_python(inner_thing) for inner_thing in v]


@convert_julia_to_python.register
def _(v: tuple):
    return tuple(convert_julia_to_python(inner_thing) for inner_thing in v)


@convert_julia_to_python.register
def _(v: dict):
    return {k: convert_julia_to_python(v) for k, v in v.items()}
