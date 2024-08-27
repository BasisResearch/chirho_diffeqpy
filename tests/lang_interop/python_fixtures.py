from os.path import (
    dirname as dn,
    join as jn
)
import juliacall
import numpy as np
from itertools import product
# Noop import to force package loading and diffeqpy julia environment activation.
from diffeqpy.de import de

jl = juliacall.Main.seval

julia_fixtures_pth = jn(dn(__file__), "julia_fixtures.jl")
jl(f"include(\"{julia_fixtures_pth}\")")


forward_eval = jl("forward_eval")
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
    # # multiple return values  # FIXME not currently supported. Requires small addition to conversion mappings.
    # (lambda x: (x + 1., x + 2.), (
    #     ((1.,), dict()),
    # )),
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
    # # np.ceil  # FIXME doesn't forward properly on symbolics
    # (np.ceil, (
    #     ((0.,), dict()),
    #     ((1.,), dict()),
    #     ((-1.,), dict()),
    # )),
    # # np.floor  # FIXME doesn't forward properly on symbolics
    # (np.floor, (
    #     ((0.,), dict()),
    #     ((1.,), dict()),
    #     ((-1.,), dict()),
    # )),
    # # np.round  # FIXME doesn't forward properly
    # (np.round, (
    #     ((0.,), dict()),
    #     ((1.,), dict()),
    #     ((-1.,), dict()),
    # )),
    # # np.trunc  # FIXME doesn't forward properly on symbolics
    # (np.trunc, (
    #     ((0.,), dict()),
    #     ((1.,), dict()),
    #     ((-1.,), dict()),
    # )),
    # # np.isnan  # FIXME 18d0j1h9 (see tag in julianumpy.internals.)
    # (np.isnan, (
    #     ((0.,), dict()),
    #     ((1.,), dict()),
    #     ((np.nan,), dict()),
    # )),
    # # np.isinf  # FIXME 18d0j1h9 (see tag in julianumpy.internals.)
    # (np.isinf, (
    #     ((0.,), dict()),
    #     ((1.,), dict()),
    #     ((np.inf,), dict()),
    # )),
    # # np.isfinite  # FIXME 18d0j1h9 (see tag in julianumpy.internals.)
    # (np.isfinite, (
    #     ((0.,), dict()),
    #     ((1.,), dict()),
    #     ((np.inf,), dict()),
    # )),
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
    # Converts the test fns/args/kwargs specified above into arrays that will op elementwise.
    for f, argskwargs in argskwargs_fns:
        new_argskwargs = []

        for (args, kwargs), shape in product(argskwargs, shapes):
            new_args = tuple(_scalar_to_array(arg, shape) for arg in args)
            new_kwargs = {k: _scalar_to_array(v, shape) for k, v in kwargs.items()}
            new_argskwargs.append((new_args, new_kwargs))

        yield f, tuple(new_argskwargs)


# Test single operations on arrays, elementwise.
single_op_array_elementwise_dunder_argskwargs_fns = list(
    _reshape_argskwargs_elementwise(single_op_scalar_dunder_argskwargs_fns)
)


# Test compound operations on arrays, elementwise.
compound_op_array_elementwise_dunder_argskwargs_fns = list(
    _reshape_argskwargs_elementwise(compound_op_scalar_dunder_fns)
)


# Test single ufuncs on arrays, elementwise.
single_op_array_elementwise_ufunc_argskwargs_fns = list(
    _reshape_argskwargs_elementwise(single_op_scalar_ufunc_fns)
)


# Test compound operations on arrays, elementwise.
compound_op_array_elementwise_ufunc_dunder_argskwargs_fns = list(
    _reshape_argskwargs_elementwise(compound_op_scalar_ufunc_dunder_fns)
)
# </Elementwise Array Ops>


# <Broadcasted Array Ops>
def _set_dim_to_1(dim, shape):
    shape = list(shape)
    shape[dim] = 1
    return tuple(shape)


def _reshape_argskwargs_broadcast(argskwargs_fns):
    # Converts the test fns/args/kwargs specified above into arrays that will broadcast.
    for f, argskwargs in argskwargs_fns:
        new_argskwargs = []

        for args, kwargs in argskwargs:
            num_args, num_kwargs = len(args), len(kwargs)
            num_inputs = num_args + num_kwargs
            if num_inputs <= 1:
                continue
            end_shape = tuple(range(2, num_inputs + 2))

            new_args = [_scalar_to_array(arg, _set_dim_to_1(i, end_shape)) for i, arg in enumerate(args)]
            new_kwargs = {k: _scalar_to_array(v, _set_dim_to_1(i + num_args, end_shape))
                          for i, (k, v) in enumerate(kwargs.items())}

            new_argskwargs.append((tuple(new_args), new_kwargs))

        if not len(new_argskwargs):
            continue

        yield f, tuple(new_argskwargs)


# Test single operations on arrays, broadcasted.
single_op_array_broadcasted_dunder_argskwargs_fns = list(
    _reshape_argskwargs_broadcast(single_op_scalar_dunder_argskwargs_fns)
)


# Test compound operations on arrays, broadcasted.
compound_op_array_broadcasted_dunder_argskwargs_fns = list(
    _reshape_argskwargs_broadcast(compound_op_scalar_dunder_fns)
)


# Test single ufuncs on arrays, broadcasted.
single_op_array_broadcasted_ufunc_argskwargs_fns = list(
    _reshape_argskwargs_broadcast(single_op_scalar_ufunc_fns)
)


# Test compound operations on arrays, broadcasted.
compound_op_array_broadcasted_ufunc_dunder_argskwargs_fns = list(
    _reshape_argskwargs_broadcast(compound_op_scalar_ufunc_dunder_fns)
)
# </Broadcasted Array Ops>

# <Reduction Ops>

#  array to reduce
_a2r = np.random.randn(3, 5)

reduction_op_array_fns = [
    # # np.max # FIXME 18d0j1h9 (see tag in julianumpy.internals.)
    # (lambda x: np.max(x, axis=0), (
    #     ((_a2r,), dict()),
    # )),
    # (lambda x: np.max(x, axis=(0, 1)), (
    #     ((_a2r,), dict()),
    # )),
    # (lambda x: np.max(x, axis=None), (
    #     ((_a2r,), dict()),
    # )),
    # # np.min # FIXME 18d0j1h9 (see tag in julianumpy.internals.)
    # (lambda x: np.min(x, axis=0), (
    #     ((_a2r,), dict()),
    # )),
    # (lambda x: np.min(x, axis=(0, 1)), (
    #     ((_a2r,), dict()),
    # )),
    # (lambda x: np.min(x, axis=None), (
    #     ((_a2r,), dict()),
    # )),
    # np.sum
    (lambda x: np.sum(x, axis=0), (
        ((_a2r,), dict()),
    )),
    (lambda x: np.sum(x, axis=(0, 1)), (
        ((_a2r,), dict()),
    )),
    (lambda x: np.sum(x, axis=None), (
        ((_a2r,), dict()),
    )),
    # np.prod
    (lambda x: np.prod(x, axis=0), (
        ((_a2r,), dict()),
    )),
    (lambda x: np.prod(x, axis=(0, 1)), (
        ((_a2r,), dict()),
    )),
    (lambda x: np.prod(x, axis=None), (
        ((_a2r,), dict()),
    )),
    # # np.mean # FIXME 18d0j1h9 (see tag in julianumpy.internals.)
    # (lambda x: np.mean(x, axis=0), (
    #     ((_a2r,), dict()),
    # )),
    # (lambda x: np.mean(x, axis=(0, 1)), (
    #     ((_a2r,), dict()),
    # )),
    # (lambda x: np.mean(x, axis=None), (
    #     ((_a2r,), dict()),
    # )),
    # # np.std # FIXME 18d0j1h9 (see tag in julianumpy.internals.)
    # (lambda x: np.std(x, axis=0), (
    #     ((_a2r,), dict()),
    # )),
    # (lambda x: np.std(x, axis=(0, 1)), (
    #     ((_a2r,), dict()),
    # )),
    # (lambda x: np.std(x, axis=None), (
    #     ((_a2r,), dict()),
    # )),
    # @ matmul operator
    (lambda x: x @ x.T, (
        ((_a2r,), dict()),
    )),
    # np.einsum
    (lambda x: np.einsum("ij->j", x), (
        ((_a2r,), dict()),
    )),
    # TODO ...
]
# </Reduction Ops>

# TODO test interop with scalars and arrays.

# TODO test interop with everything, scalars, arrays, ufuncs, etc.

# TODO test slicing.
