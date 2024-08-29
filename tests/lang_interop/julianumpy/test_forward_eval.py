import numpy as np
import pytest

# Unused import to register the conversion overloads.
import chirho_diffeqpy.lang_interop.julianumpy
from chirho_diffeqpy.lang_interop import callable_from_julia

from ..python_fixtures import (
    compound_op_array_broadcasted_dunder_argskwargs_fns,
    compound_op_array_broadcasted_ufunc_dunder_argskwargs_fns,
    compound_op_array_elementwise_dunder_argskwargs_fns,
    compound_op_array_elementwise_ufunc_dunder_argskwargs_fns,
    compound_op_scalar_dunder_fns,
    compound_op_scalar_ufunc_dunder_fns,
    forward_eval,
    reduction_op_array_fns,
    single_op_array_broadcasted_dunder_argskwargs_fns,
    single_op_array_broadcasted_ufunc_argskwargs_fns,
    single_op_array_elementwise_dunder_argskwargs_fns,
    single_op_array_elementwise_ufunc_argskwargs_fns,
    single_op_scalar_dunder_argskwargs_fns,
    single_op_scalar_ufunc_fns,
)


# Common test logic here, so-as to separating out tests. This helps pytest more clearly indicate what is broken.
def _forward_eval_test(f_args_kwargs):
    f, argskwargs = f_args_kwargs

    f_from_jl = callable_from_julia(out_dtype=np.float64)(f)
    f_from_py = f

    for args, kwargs in argskwargs:
        # Ensure julia return and python return are the same.
        py_val = f_from_py(*args, **kwargs)
        jl_val = forward_eval(f_from_jl, *args, **kwargs)

        # TODO TODO WIP this is weird, we want to compare the raw python eval (python inputs, python outputs) to the
        #  eval where the python function is passed to julia and evaluated there, and then returned here. The tricky
        #  piece is that the thing returned from julia is going to be...kind of its own thing.
        # TODO TODO WIP also, testing forward eval is kinda moot tbh? For the backend, what we really care about is
        #  the symbolic forward eval and symbolic forward diff eval.

        assert np.allclose(py_val, jl_val)


# <editor-fold desc="Scalar Tests">
# TODO put in separate file so the pytest feedback is more organized.
@pytest.mark.parametrize("f_args_kwargs", single_op_scalar_dunder_argskwargs_fns)
def test_forward_eval__single_op_scalar_dunder_argskwargs_test_funcs(f_args_kwargs):
    _forward_eval_test(f_args_kwargs)


@pytest.mark.parametrize("f_args_kwargs", compound_op_scalar_dunder_fns)
def test_forward_eval__compound_op_scalar_dunder_test_funcs(f_args_kwargs):
    _forward_eval_test(f_args_kwargs)


@pytest.mark.parametrize("f_args_kwargs", single_op_scalar_ufunc_fns)
def test_forward_eval__single_op_scalar_ufunc_test_funcs(f_args_kwargs):
    _forward_eval_test(f_args_kwargs)


@pytest.mark.parametrize("f_args_kwargs", compound_op_scalar_ufunc_dunder_fns)
def test_forward_eval__compound_op_scalar_ufunc_test_funcs(f_args_kwargs):
    _forward_eval_test(f_args_kwargs)


# </editor-fold>


# <editor-fold desc="Elementwise Array Tests">
# TODO put in separate file so the pytest feedback is more organized.
@pytest.mark.parametrize(
    "f_args_kwargs", single_op_array_elementwise_dunder_argskwargs_fns
)
def test_forward_eval__single_op_array_elementwise_dunder_argskwargs_test_funcs(
    f_args_kwargs,
):
    _forward_eval_test(f_args_kwargs)


@pytest.mark.parametrize(
    "f_args_kwargs", compound_op_array_elementwise_dunder_argskwargs_fns
)
def test_forward_eval__compound_op_array_elementwise_dunder_argskwargs_test_funcs(
    f_args_kwargs,
):
    _forward_eval_test(f_args_kwargs)


@pytest.mark.parametrize(
    "f_args_kwargs", single_op_array_elementwise_ufunc_argskwargs_fns
)
def test_forward_eval__single_op_array_elementwise_ufunc_argskwargs_test_funcs(
    f_args_kwargs,
):
    _forward_eval_test(f_args_kwargs)


@pytest.mark.parametrize(
    "f_args_kwargs", compound_op_array_elementwise_ufunc_dunder_argskwargs_fns
)
def test_forward_eval__compound_op_array_elementwise_ufunc_argskwargs_test_funcs(
    f_args_kwargs,
):
    _forward_eval_test(f_args_kwargs)


# </editor-fold>


# <editor-fold desc="Broadcasted Array Tests">
# TODO put in separate file so the pytest feedback is more organized.
@pytest.mark.parametrize(
    "f_args_kwargs", single_op_array_broadcasted_dunder_argskwargs_fns
)
def test_forward_eval__single_op_array_broadcasted_dunder_argskwargs_test_funcs(
    f_args_kwargs,
):
    _forward_eval_test(f_args_kwargs)


@pytest.mark.parametrize(
    "f_args_kwargs", compound_op_array_broadcasted_dunder_argskwargs_fns
)
def test_forward_eval__compound_op_array_broadcasted_dunder_argskwargs_test_funcs(
    f_args_kwargs,
):
    _forward_eval_test(f_args_kwargs)


@pytest.mark.parametrize(
    "f_args_kwargs", single_op_array_broadcasted_ufunc_argskwargs_fns
)
def test_forward_eval__single_op_array_broadcasted_ufunc_argskwargs_test_funcs(
    f_args_kwargs,
):
    _forward_eval_test(f_args_kwargs)


@pytest.mark.parametrize(
    "f_args_kwargs", compound_op_array_broadcasted_ufunc_dunder_argskwargs_fns
)
def test_forward_eval__compound_op_array_broadcasted_ufunc_argskwargs_test_funcs(
    f_args_kwargs,
):
    _forward_eval_test(f_args_kwargs)


# </editor-fold>


# <editor-fold desc="Reduction Array Tests">
# TODO put in separate file so the pytest feedback is more organized.
@pytest.mark.parametrize("f_args_kwargs", reduction_op_array_fns)
def test_forward_eval__reduction_op_array_test_funcs(f_args_kwargs):
    _forward_eval_test(f_args_kwargs)


# </editor-fold>
