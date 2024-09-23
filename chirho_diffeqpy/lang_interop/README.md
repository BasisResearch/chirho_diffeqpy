# Language Interoperation

This module contains language interoperation backends for julia-python interoperation. Syntactically, a python function 
decorated with `@lang_interop.callable_from_julia` will have its julia arguments converted by 
`lang_interop.ops.julia_to_python` before being passed to the python function. The return value will be converted by
`lang_interop.ops.python_to_julia` before being passed back to julia. A backend must define a set of type-dispatched
conversions for these operations, and will typically wrap the julia objects to be compatible with e.g. `numpy` or 
`torch` functions. See the `julianumpy` backend for an example.


Importantly, this abstraction actually handles `juliacall` objects that are themselves wrapped julia objects. Future
work could explore the possibility of discarding the conversion abstractions defined here in favor of using built in,
extensible `juliacall` conversion mechanisms directly.
