import numpy as np
import pytest

# Unused import to register the conversion overloads.
import chirho_diffeqpy.lang_interop.julianumpy  # noqa: F401
from chirho_diffeqpy.lang_interop import callable_from_julia

from ..python_fixtures import (
    compound_op_array_broadcasted_dunder_argskwargs_fns,
    compound_op_array_broadcasted_ufunc_dunder_argskwargs_fns,
    compound_op_array_elementwise_dunder_argskwargs_fns,
    compound_op_array_elementwise_ufunc_dunder_argskwargs_fns,
    compound_op_scalar_dunder_fns,
    compound_op_scalar_ufunc_dunder_fns,
    reduction_op_array_fns,
    single_op_array_broadcasted_dunder_argskwargs_fns,
    single_op_array_broadcasted_ufunc_argskwargs_fns,
    single_op_array_elementwise_dunder_argskwargs_fns,
    single_op_array_elementwise_ufunc_argskwargs_fns,
    single_op_scalar_dunder_argskwargs_fns,
    single_op_scalar_ufunc_fns,
    symbolically_compile_function,
)


# Common test logic here, so-as to separating out tests. This helps pytest more clearly indicate what is broken.
def _symbolic_forward_eval_test(f_args_kwargs):
    f, argskwargs = f_args_kwargs

    f_from_jl = callable_from_julia()(f)
    f_from_py = f

    for args, kwargs in argskwargs:
        # Ensure julia return and python return are the same.

        # python in, python exec, python out
        py_val = f_from_py(*args, **kwargs)
        # python in, convert to julia, julia exec, convert back to python, python out
        compiled_fn = symbolically_compile_function(
            f_from_jl,
            *args,
            out_shape=py_val.shape if isinstance(py_val, np.ndarray) else tuple(),
            **kwargs
        )

        # Now go through and arbitrarily change all the args and kwargs to make sure that the compiled_fn is
        #  general wrt the inputs.
        args = tuple(arg + 1.0 for arg in args)
        kwargs = {k: v + 1.0 for k, v in kwargs.items()}

        py_val = f_from_py(*args, **kwargs)
        jl_val = compiled_fn(*args, **kwargs)

        assert np.allclose(py_val, jl_val)


# <editor-fold desc="Scalar Tests">
# TODO put in separate file so the pytest feedback is more organized.
@pytest.mark.parametrize("f_args_kwargs", single_op_scalar_dunder_argskwargs_fns)
def test_forward_eval__single_op_scalar_dunder_argskwargs_test_funcs(f_args_kwargs):
    _symbolic_forward_eval_test(f_args_kwargs)


@pytest.mark.parametrize("f_args_kwargs", compound_op_scalar_dunder_fns)
def test_forward_eval__compound_op_scalar_dunder_test_funcs(f_args_kwargs):
    _symbolic_forward_eval_test(f_args_kwargs)


@pytest.mark.parametrize("f_args_kwargs", single_op_scalar_ufunc_fns)
def test_forward_eval__single_op_scalar_ufunc_test_funcs(f_args_kwargs):
    _symbolic_forward_eval_test(f_args_kwargs)


@pytest.mark.parametrize("f_args_kwargs", compound_op_scalar_ufunc_dunder_fns)
def test_forward_eval__compound_op_scalar_ufunc_test_funcs(f_args_kwargs):
    _symbolic_forward_eval_test(f_args_kwargs)


# </editor-fold>


# <editor-fold desc="Elementwise Array Ops">
# TODO put in separate file so the pytest feedback is more organized.
@pytest.mark.parametrize(
    "f_args_kwargs", single_op_array_elementwise_dunder_argskwargs_fns
)
def test_forward_eval__single_op_array_elementwise_dunder_argskwargs_test_funcs(
    f_args_kwargs,
):
    _symbolic_forward_eval_test(f_args_kwargs)


@pytest.mark.parametrize(
    "f_args_kwargs", compound_op_array_elementwise_dunder_argskwargs_fns
)
def test_forward_eval__compound_op_array_elementwise_dunder_argskwargs_test_funcs(
    f_args_kwargs,
):
    _symbolic_forward_eval_test(f_args_kwargs)


@pytest.mark.parametrize(
    "f_args_kwargs", single_op_array_elementwise_ufunc_argskwargs_fns
)
def test_forward_eval__single_op_array_elementwise_ufunc_argskwargs_test_funcs(
    f_args_kwargs,
):
    _symbolic_forward_eval_test(f_args_kwargs)


@pytest.mark.parametrize(
    "f_args_kwargs", compound_op_array_elementwise_ufunc_dunder_argskwargs_fns
)
def test_forward_eval__compound_op_array_elementwise_ufunc_argskwargs_test_funcs(
    f_args_kwargs,
):
    _symbolic_forward_eval_test(f_args_kwargs)


# </editor-fold>


# <editor-fold desc="Broadcasted Array Ops">
# TODO put in separate file so the pytest feedback is more organized.
@pytest.mark.parametrize(
    "f_args_kwargs", single_op_array_broadcasted_dunder_argskwargs_fns
)
def test_forward_eval__single_op_array_broadcasted_dunder_argskwargs_test_funcs(
    f_args_kwargs,
):
    _symbolic_forward_eval_test(f_args_kwargs)


@pytest.mark.parametrize(
    "f_args_kwargs", compound_op_array_broadcasted_dunder_argskwargs_fns
)
def test_forward_eval__compound_op_array_broadcasted_dunder_argskwargs_test_funcs(
    f_args_kwargs,
):
    _symbolic_forward_eval_test(f_args_kwargs)


@pytest.mark.parametrize(
    "f_args_kwargs", single_op_array_broadcasted_ufunc_argskwargs_fns
)
def test_forward_eval__single_op_array_broadcasted_ufunc_argskwargs_test_funcs(
    f_args_kwargs,
):
    _symbolic_forward_eval_test(f_args_kwargs)


@pytest.mark.parametrize(
    "f_args_kwargs", compound_op_array_broadcasted_ufunc_dunder_argskwargs_fns
)
def test_forward_eval__compound_op_array_broadcasted_ufunc_dunder_argskwargs_test_funcs(
    f_args_kwargs,
):
    _symbolic_forward_eval_test(f_args_kwargs)


# </editor-fold>


# <editor-fold desc="Reduction Array Ops">
# TODO put in separate file so the pytest feedback is more organized.
@pytest.mark.parametrize("f_args_kwargs", reduction_op_array_fns)
def test_forward_eval__reduction_op_array_test_funcs(f_args_kwargs):
    _symbolic_forward_eval_test(f_args_kwargs)


# </editor-fold>
