import numpy as np
import pytest
from ..python_fixtures import (
    symbolic_forward_eval,
    single_op_scalar_dunder_argskwargs_fns,
    compound_op_scalar_dunder_fns,
    single_op_scalar_ufunc_fns,
    compound_op_scalar_ufunc_dunder_fns,
    single_op_array_elementwise_dunder_argskwargs_fns,
    compound_op_array_elementwise_dunder_argskwargs_fns,
    single_op_array_elementwise_ufunc_argskwargs_fns,
    compound_op_array_elementwise_ufunc_dunder_argskwargs_fns,
)
from chirho_diffeqpy.lang_interop import callable_from_julia
# Unused import to register the conversion overloads.
import chirho_diffeqpy.lang_interop.julianumpy


# Common test logic here, so-as to separating out tests. This helps pytest more clearly indicate what is broken.
def _symbolic_forward_eval_test(f_args_kwargs):
    f, argskwargs = f_args_kwargs

    f_from_jl = callable_from_julia(f, out_dtype=np.float64)
    f_from_py = f

    for args, kwargs in argskwargs:
        # Ensure julia return and python return are the same.

        # python in, python exec, python out
        py_val = f_from_py(*args, **kwargs)
        # python in, convert to julia, julia exec, convert back to python, python out
        jl_val = symbolic_forward_eval(f_from_jl, *args, **kwargs)

        assert np.allclose(py_val, jl_val)


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


@pytest.mark.parametrize("f_args_kwargs", single_op_array_elementwise_dunder_argskwargs_fns)
def test_forward_eval__single_op_array_elementwise_dunder_argskwargs_test_funcs(f_args_kwargs):
    _symbolic_forward_eval_test(f_args_kwargs)


@pytest.mark.parametrize("f_args_kwargs", compound_op_array_elementwise_dunder_argskwargs_fns)
def test_forward_eval__compound_op_array_elementwise_dunder_argskwargs_test_funcs(f_args_kwargs):
    _symbolic_forward_eval_test(f_args_kwargs)


@pytest.mark.parametrize("f_args_kwargs", single_op_array_elementwise_ufunc_argskwargs_fns)
def test_forward_eval__single_op_array_elementwise_ufunc_argskwargs_test_funcs(f_args_kwargs):
    _symbolic_forward_eval_test(f_args_kwargs)


@pytest.mark.parametrize("f_args_kwargs", compound_op_array_elementwise_ufunc_dunder_argskwargs_fns)
def test_forward_eval__compound_op_array_elementwise_ufunc_argskwargs_test_funcs(f_args_kwargs):
    _symbolic_forward_eval_test(f_args_kwargs)
