from typing import Optional
import numpy as np
from functools import singledispatch
from juliacall import Main as jl

# TODO refactor the comments here. This solves two problems:
#  1) julia things have all dunders defined, which means that they isinstance every base class in python, even
#      though most of the time those dunders are not implemented. This is very confusing for python (and numpy)
#      and eg causes numpy to iterate a julia scalar and get a >32 dim array.
#  2) numpy ufuncs don't work on julia objects, so we need to forward those to their respective julia functions.
#  The point here is to be able to write a numpy function, and then give that to julia to execute
#   from within julia code.
#  Then most all other shenanigans essentially ensures that everything stays properly wrapped (from numpy's perspective)
#   the entire time it's all going through the numpy code.


class _DunderedJuliaThingWrapper:
    """
    Handles just the dunder forwarding to the undelrying julia thing. Beyond a separation of concerns, this is
    a separate class because we need to be able to cast back into something that doesn't have custom __array_ufunc__
    behavior. See _default_ufunc below for more details on this nuance.
    """

    def __init__(self, julia_thing):
        self.julia_thing = julia_thing

    @classmethod
    def _forward_dunders(cls):

        # Forward all the math related dunder methods to the underlying julia thing.
        dunders = [
            '__abs__',
            '__add__',
            # FIXME 18d0j1h9 python sometimes expects not a JuliaThingWrapper from __bool__, what do e.g. julia
            #  symbolics expect?
            '__bool__',
            '__ceil__',
            '__eq__',  # FIXME 18d0j1h9
            '__float__',
            '__floor__',
            '__floordiv__',
            '__ge__',  # FIXME 18d0j1h9
            '__gt__',  # FIXME 18d0j1h9
            '__invert__',
            '__le__',  # FIXME 18d0j1h9
            '__lshift__',
            '__lt__',  # FIXME 18d0j1h9
            '__mod__',
            '__mul__',
            '__ne__',
            '__neg__',
            '__or__',  # FIXME 18d0j1h9
            '__pos__',
            '__pow__',
            '__radd__',
            '__rand__',  # FIXME 18d0j1h9 (also, where is __and__?)
            '__reversed__',
            '__rfloordiv__',
            '__rlshift__',
            '__rmod__',
            '__rmul__',
            '__ror__',
            '__round__',
            '__rpow__',
            '__rrshift__',
            '__rshift__',
            '__rsub__',
            '__rtruediv__',
            '__rxor__',
            '__sub__',
            '__truediv__',
            '__trunc__',
            '__xor__',  # FIXME 18d0j1h9
        ]

        for method_name in dunders:
            cls._make_dunder(method_name)

    @classmethod
    def _make_dunder(cls, method_name):
        """
        Automate the definition of dunder methods involving the underlying julia things. Note that just intercepting
        getattr doesn't work here because dunder method calls skip getattr, and getattribute is fairly complex
        to work with.
        """

        def dunder(self: _DunderedJuliaThingWrapper, *args):
            # Retrieve the underlying dunder method of the julia thing.
            method = getattr(self.julia_thing, method_name)

            if not args:
                # E.g. __neg__, __pos__, __abs__ don't have an "other"
                result = method()

                if result is NotImplemented:
                    raise NotImplementedError(f"Operation {method_name} is not implemented for {self.julia_thing}.")
            else:
                if len(args) != 1:
                    raise ValueError("Only one argument is supported for automated dunder method dispatch.")
                other, = args

                if isinstance(other, np.ndarray):
                    if other.ndim == 0:
                        # In certain cases, that TODO need to be sussed out (maybe numpy internal nuance) the
                        #  julia_thing is a scalar array of a JuliaThingWrapper, so we need to further unwrap the
                        #  scalar array to get at the JuliaThingWrapper (and, in turn, the julia_thing).
                        other = other.item()
                    else:
                        # Wrap self in an array and recurse back through numpy broadcasting. This is required when a
                        #  JuliaThingWrapper "scalar" is involved in an operation with a numpy array on the right.
                        scalar_array_self = np.array(self)
                        scalar_array_self_attr = getattr(scalar_array_self, method_name)
                        return scalar_array_self_attr(other)

                # Extract the underlying julia thing.
                if isinstance(other, _DunderedJuliaThingWrapper):
                    other = other.julia_thing

                # Perform the operation using the corresponding method of the Julia object
                result = method(other)

                if result is NotImplemented:
                    raise NotImplementedError(f"Operation {method_name} is not implemented for"
                                              f" {self.julia_thing} and {other}.")

            # Rewrap the return.
            return JuliaThingWrapper(result)

        setattr(cls, method_name, dunder)


# noinspection PyProtectedMember
_DunderedJuliaThingWrapper._forward_dunders()


class JuliaThingWrapper(_DunderedJuliaThingWrapper):
    """
    This wrapper just acts as a pass-through to the julia object, but obscures the underlying memory buffer of the
    julia thing (realvalue, symbolic, dual number, etc.). This prevents numpy from introspecting the julia thing as a
    sequence with a large number of dimensions (exceeding the ndim 32 limit). Unfortunately, even with a dtype of
    np.object_, this introspection still occurs. The issue of casting to a numpy array can also be addressed by first
    creating an empty array of dtype object, and then filling it with the julia thing (as occurs in unwrap_array
    below), but this fails to generalize well in cases where numpy is doing the casting itself. As of now, this seems
    the most robust solution.

    Note that numpy arrays of objects will, internally, use the dunder math methods of the objects they contain when
    performing math operations. This is not fast, but for our purposes is fine b/c the main application here involves
    julia symbolics only during jit compilation. As such, the point of this class is to wrap scalar valued julia things
    only so that we can use numpy arrays of julia things.

    This class also handles the forwarding of numpy universal functions like sin, exp, log, etc. to the corresopnding
    julia version. See __array_ufunc__ for more details.
    """

    @staticmethod
    def wrap_array(arr: np.ndarray):
        regular_array = np.vectorize(JuliaThingWrapper)(arr)
        # Because we need to forward numpy ufuncs to julia,
        return regular_array.view(_JuliaThingWrapperArray)

    @staticmethod
    def unwrap_array(arr: np.ndarray, out: Optional[np.ndarray] = None, out_dtype: Optional = None):

        if out is not None and out_dtype is not None:
            raise ValueError("Only one of out and out_dtype can be specified. If you pass an out array, the dtype"
                             " of the unwrapped array will be inferred from the dtype of the out array.")

        if out is None and out_dtype is None:
            raise ValueError("Either out or out_dtype must be specified.")

        # As discussed in docstring, we cannot simply vectorize a deconstructor because numpy will try to internally
        #  cast the unwrapped_julia things into an array, which fails due to introspection triggering the ndim 32 limit.
        # Instead, we have to manually assign each element of the array. This is slow, but only occurs during jit
        #  compilation for our use case.
        if out is None:
            print("making out array")
            out = np.empty(arr.shape, dtype=out_dtype)

            print("out.shape", out.shape)
            print("out.dtype", out.dtype)

        for idx, v in np.ndenumerate(arr):
            out[idx] = v.julia_thing
        print("----finished----")
        return out

    def __repr__(self):
        return f"JuliaThingWrapper({self.julia_thing})"

    def _jl_ufunc(self, ufunc):
        # Try to grab something from the
        ufunc_name = ufunc.__name__
        try:
            jlfunc = getattr(jl, ufunc_name)
        except AttributeError:
            # when __array_ufunc__ fails to resolve, it returns NotImplemented, so this follows that pattern.
            return NotImplemented

        result = jlfunc(self.julia_thing)

        return JuliaThingWrapper(result)

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        """
        Many numpy functions (like sin, exp, log, etc.) are so-called "universal functions" that don't correspond to
        a standard dunder method. To handle these, we need to dispatch the julia version of the function on the
        underlying julia thing. This is done by overriding __array_ufunc__ and forwarding the call to the jl
        function operating on the underlying julia thing, assuming that the corresponding dunder method hasn't
        already been defined.
        """

        # First try to evaluate the default ufunc (this will dispatch first to dunder methods, like __abs__).
        ret = _default_ufunc(ufunc, method, *args, **kwargs)
        if ret is not NotImplemented:
            return ret

        # Otherwise, try to dispatch the ufunc to the underlying julia thing.
        return self._jl_ufunc(ufunc)


class _JuliaThingWrapperArray(np.ndarray):
    """
    Subclassing the numpy array in order to translate ufunc calls to julia equivalent calls at the array level (
    rather than the element level). This is required because numpy doesn't defer to the __array_ufunc__ method of the
    underlying object for arrays of dtype object.
    """

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        # First, try to resolve the ufunc in the standard manner (this will dispatch first to dunder methods).
        ret = _default_ufunc(ufunc, method, *args, **kwargs)
        if ret is not NotImplemented:
            return ret

        # Otherwise, because numpy doesn't defer to the ufuncs of the underlying objects when being called on an array
        # of objects (unlike how it behaves with dunder-definable ops), iterate manually and do so here.
        result = _JuliaThingWrapperArray(self.shape, dtype=object)

        for idx, v in np.ndenumerate(self):
            assert isinstance(v, JuliaThingWrapper)  # TODO value error instead
            result[idx] = v._jl_ufunc(ufunc)

        return result


def _default_ufunc(ufunc, method, *args, **kwargs):
    f = getattr(ufunc, method)

    # Numpy's behavior changes if __array_ufunc__ is defined at all, i.e. a super() call is insufficient to
    #  capture the default behavior as if no __array_ufunc__ were involved. The way to do this is to create
    #  a standard np.ndarray view of the underlying memory, and then call the ufunc on that.
    nargs = (_cast_as_lacking_array_ufunc(x) for x in args)

    try:
        ret = f(*nargs, **kwargs)
        return _cast_as_having_array_func(ret)
    except TypeError as e:
        # If the exception reports anything besides non-implementation of the ufunc, then re-raise.
        if f"no callable {ufunc.__name__} method" not in str(e):
            raise
        # Otherwise, just return NotImplemented in keeping with standard __array_ufunc__ behavior.
        else:
            return NotImplemented


# These functions handle casting back and forth from entities that have custom behavior for numpy ufuncs, and those
# that don't.
@singledispatch
def _cast_as_lacking_array_ufunc(v):
    return v


@_cast_as_lacking_array_ufunc.register
def _(v: _JuliaThingWrapperArray):
    return v.view(np.ndarray)


@_cast_as_lacking_array_ufunc.register
def _(v: JuliaThingWrapper):
    return _DunderedJuliaThingWrapper(v.julia_thing)


@singledispatch
def _cast_as_having_array_func(v):
    return v


@_cast_as_having_array_func.register
def _(v: np.ndarray):
    return v.view(_JuliaThingWrapperArray)


@_cast_as_having_array_func.register
def _(v: _DunderedJuliaThingWrapper):
    return JuliaThingWrapper(v.julia_thing)
