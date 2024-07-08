from os.path import (
    dirname as dn,
    join as jn
)
import juliacall
import numpy as np
from itertools import product

jl = juliacall.Main.seval

julia_fixtures_pth = jn(dn(__file__), "julia_fixtures.jl")
jl(f"include(\"{julia_fixtures_pth}\")")


forward_eval = jl("forward_eval")
symbolic_forward_eval = jl("symbolic_forward_eval")
symbolically_compile_function = jl("symbolically_compile_function")


# Simple functions with single operations and scalars, primarily testing that args/kwargs wrap correctly.
single_op_scalar_dunder_argskwargs_fns = [
    # negation.
    (lambda x: -x, (
        ((1.,), dict()),
        ((-1.,), dict()),
    )),
    # constant other, pow
    (lambda x: x ** 2., (
        ((2.,), dict()),  # one arg
        (tuple(), dict(x=-2.)),  # one kwarg
    )),
    # variable other, addition, testing kwarg/args as well.
    (lambda x, y: x + y, (
        ((1., 2.), dict()),  # two args
        ((-1.,), dict(y=1.)),  # one arg one kwarg
        (tuple(), dict(x=-3., y=0.)),  # two kwargs
    )),
    # Addition again but with one default kwarg.
    (lambda x, y=1.: x + y, (
        ((1.,), dict()),  # one arg
        ((-1.,), dict(y=1.)),  # one arg one kwarg
        (tuple(), dict(x=-3.)),  # one non-default kwarg
    )),
    # Addition again but with two default kwargs.
    (lambda x=1., y=1.: x + y, (
        (tuple(), dict()),  # no args
        ((-1.,), dict(y=1.)),  # one arg one kwarg
        (tuple(), dict(x=-3.)),  # one non-default kwarg
    )),
    # variable other, multiplication, just testing op.
    (lambda x, y: x * y, (
        ((1., 2.), dict()),
    )),
    # variable other, division, just testing op.
    (lambda x, y: x / y, (
        ((1., 2.), dict()),
    )),
    # variable other, subtraction, just testing op.
    (lambda x, y: x ** y, (
        ((1., 2.), dict()),
    )),
    # multiple return values
    (lambda x: (x + 1., x + 2.), (
        ((1.,), dict()),
    )),
]


# Tests of compound functions with 2-5 operations on scalar arguments. No need to test different arg/kwarg configs.
compound_op_scalar_dunder_fns = [
    # constant other, pow, addition
    (lambda x: (x + 1) ** 2., (
        ((2.,), dict()),
    )),
    # variable other, addition, pow
    (lambda x, y: x ** y + y, (
        ((2., 3.), dict()),
    )),
    # variable other, addition, pow, multiplication
    (lambda x, y, z: (x ** y + y) * z, (
        ((2., 3., 4.), dict()),
    )),
    # variable other, addition, pow, multiplication, division
    (lambda x, y, z, w: ((x ** y + y) * z) / w, (
        ((2., 3., 4., 5.), dict()),
    )),
    # variable other, addition, pow, multiplication, division, subtraction
    (lambda x, y, z, w, v: (((x ** y + y) * z) / w) - v, (
        ((2., 3., 4., 5., 6.), dict()),
        ((-2., 3., -4., -5., 6.), dict()),
        ((-2., -3., -4., -5., -6.), dict()),
    )),
]


# Tests of single ufuncs on scalars, use a few different values of arguments for each.
# This test set should cover anything we might use, while other ufunc coverage is more about compound considerations.
# noinspection DuplicatedCode
single_op_scalar_ufunc_fns = [
    # np.exp
    (np.exp, (
        ((0.,), dict()),
        ((1.,), dict()),
        ((-1.,), dict()),
        ((2.,), dict()),
        ((-2.,), dict()),
    )),
    # np.log
    (np.log, (
        ((1.,), dict()),
        ((np.e,), dict()),
        ((np.e ** 2.,), dict()),
    )),
    # np.sin
    (np.sin, (
        ((0.,), dict()),
        ((np.pi / 2.,), dict()),
        ((np.pi,), dict()),
        ((-np.pi / 2.,), dict()),
        ((-np.pi,), dict()),
    )),
    # np.cos
    (np.cos, (
        ((0.,), dict()),
        ((np.pi / 2.,), dict()),
        ((np.pi,), dict()),
        ((-np.pi / 2.,), dict()),
        ((-np.pi,), dict()),
    )),
    # np.tan
    (np.tan, (
        ((0.,), dict()),
        ((np.pi / 4.,), dict()),
        ((-np.pi / 4.,), dict()),
    )),
    # # np.arcsin  # FIXME doesn't forward properly
    # (np.arcsin, (
    #     ((0.,), dict()),
    #     ((1.,), dict()),
    #     ((-1.,), dict()),
    # )),
    # # np.arccos  # FIXME doesn't forward properly
    # (np.arccos, (
    #     ((0.,), dict()),
    #     ((1.,), dict()),
    #     ((-1.,), dict()),
    # )),
    # # np.arctan  # FIXME doesn't forward properly
    # (np.arctan, (
    #     ((0.,), dict()),
    #     ((1.,), dict()),
    #     ((-1.,), dict()),
    # )),
    # np.sinh
    (np.sinh, (
        ((0.,), dict()),
        ((1.,), dict()),
        ((-1.,), dict()),
    )),
    # np.cosh
    (np.cosh, (
        ((0.,), dict()),
        ((1.,), dict()),
        ((-1.,), dict()),
    )),
    # np.tanh
    (np.tanh, (
        ((0.,), dict()),
        ((1.,), dict()),
        ((-1.,), dict()),
    )),
    # # np.arcsinh  # FIXME doesn't forward properly
    # (np.arcsinh, (
    #     ((0.,), dict()),
    #     ((1.,), dict()),
    #     ((-1.,), dict()),
    # )),
    # # np.arccosh  # FIXME doesn't forward properly
    # (np.arccosh, (
    #     ((1.,), dict()),
    #     ((2.,), dict()),
    # )),
    # # np.arctanh  # FIXME doesn't forward properly
    # (np.arctanh, (
    #     ((0.,), dict()),
    #     ((0.5,), dict()),
    #     ((-0.5,), dict()),
    # )),
    # np.sqrt
    (np.sqrt, (
        ((0.,), dict()),
        ((1.,), dict()),
        ((2.,), dict()),
    )),
    # np.abs
    (np.abs, (
        ((0.,), dict()),
        ((1.,), dict()),
        ((-1.,), dict()),
    )),
    # np.ceil
    (np.ceil, (
        ((0.,), dict()),
        ((1.,), dict()),
        ((-1.,), dict()),
    )),
    # np.floor
    (np.floor, (
        ((0.,), dict()),
        ((1.,), dict()),
        ((-1.,), dict()),
    )),
    # # np.round  # FIXME doesn't forward properly
    # (np.round, (
    #     ((0.,), dict()),
    #     ((1.,), dict()),
    #     ((-1.,), dict()),
    # )),
    # np.trunc
    (np.trunc, (
        ((0.,), dict()),
        ((1.,), dict()),
        ((-1.,), dict()),
    )),
    # # np.isnan  # FIXME doesn't forward properly
    # (np.isnan, (
    #     ((0.,), dict()),
    #     ((1.,), dict()),
    #     ((np.nan,), dict()),
    # )),
    # # np.isinf  # FIXME doesn't forward properly
    # (np.isinf, (
    #     ((0.,), dict()),
    #     ((1.,), dict()),
    #     ((np.inf,), dict()),
    # )),
    # # np.isfinite  # FIXME doesn't forward properly
    # (np.isfinite, (
    #     ((0.,), dict()),
    #     ((1.,), dict()),
    #     ((np.inf,), dict()),
    # )),
    # # np.max  # FIXME doesn't forward properly
    # (np.max, (
    #     ((0., 1.), dict()),
    #     ((1., 0.), dict()),
    #     ((-1., 0.), dict()),
    # )),
    # # np.min  # FIXME doesn't forward properly
    # (np.min, (
    #     ((0., 1.), dict()),
    #     ((1., 0.), dict()),
    #     ((-1., 0.), dict()),
    # )),
    # # np.sum  # FIXME doesn't forward properly
    # (np.sum, (
    #     ((0., 1.), dict()),
    #     ((1., 0.), dict()),
    #     ((-1., 0.), dict()),
    # )),
    # # np.prod  # FIXME doesn't forward properly
    # (np.prod, (
    #     ((0., 1.), dict()),
    #     ((1., 2.), dict()),
    #     ((-1., -1.), dict()),
    # )),
    # # np.mean  # FIXME doesn't forward properly
    # (np.mean, (
    #     ((0., 1.), dict()),
    #     ((1., 0.), dict()),
    #     ((-1., 0.), dict()),
    # )),
    # # np.std  # FIXME doesn't forward properly
    # (np.std, (
    #     ((0., 1.), dict()),
    #     ((1., 0.), dict()),
    #     ((-1., 0.), dict()),
    # )),
    # TODO ...
]


# Tests combining ufuncs and dunders on scalars. We don't necessarily need full coverage of ufuncs here.
#  should have significant variety.
# noinspection DuplicatedCode
compound_op_scalar_ufunc_dunder_fns = [
    # gaussian pdf
    (lambda x, mu, sigma: np.exp(-0.5 * ((x - mu) / sigma) ** 2.) / (sigma * np.sqrt(2. * np.pi)), (
        ((0., 0., 1.), dict()),
        ((1., 0., 1.), dict()),
        ((0., 1., 1.), dict()),
        ((1., 1., 1.), dict()),
    )),
    # polynomial
    (lambda x, a, b, c: a * x ** 2. + b * x + c, (
        ((0., 1., 2., 3.), dict()),
        ((1., 1., 1., 1.), dict()),
        ((-1., 1., -1., 1.), dict()),
    )),
    # logistic
    (lambda x, a, b, c: c / (1. + np.exp(-a * (x - b))), (
        ((0., 1., 2., 3.), dict()),
        ((1., 1., 1., 1.), dict()),
        ((-1., 1., -1., 1.), dict()),
    ))
]


def _scalar_to_array(scalar, shape):
    return np.full(shape, scalar)


# <Elementwise Array Ops>
def _reshape_argskwargs_elementwise(argskwargs_fns, shapes=((2,), (2, 1), (2, 2), (3, 2))):
    for f, argskwargs in argskwargs_fns:
        new_argskwargs = []

        for (args, kwargs), shape in product(argskwargs, shapes):
            new_args = tuple(_scalar_to_array(arg, shape) for arg in args)
            new_kwargs = {k: _scalar_to_array(v, shape) for k, v in kwargs.items()}
            new_argskwargs.append((new_args, new_kwargs))

        yield f, tuple(new_argskwargs)


# Test single operations on arrays, elementwise.
single_op_array_elementwise_dunder_argskwargs_fns = _reshape_argskwargs_elementwise(single_op_scalar_dunder_argskwargs_fns)


# Test compound operations on arrays, elementwise.
compound_op_array_elementwise_dunder_argskwargs_fns = _reshape_argskwargs_elementwise(compound_op_scalar_dunder_fns)


# Test single ufuncs on arrays, elementwise.
single_op_array_elementwise_ufunc_argskwargs_fns = _reshape_argskwargs_elementwise(single_op_scalar_ufunc_fns)


# Test compound operations on arrays, elementwise.
compound_op_array_elementwise_ufunc_dunder_argskwargs_fns = _reshape_argskwargs_elementwise(compound_op_scalar_ufunc_dunder_fns)
# </Elementwise Array Ops>

# TODO broadcasting and matrices

# TODO interop with scalars and arrays.

# TODO interop with everything, scalars, arrays, ufuncs, etc.
