# Must precede even indirect torch imports to preventshape segfault.
import juliacall  # noqa: F401
import pytest
import torch
from fixtures import ab_xy_dynfuncs, ab_xy_prior

from chirho_diffeqpy.internals import pre_broadcast_initial_state


def exec_numpy_from_torch(func, state, params):
    state = {k: v.numpy() for k, v in state.items()}
    state["t"] = 0.0
    params = {k: v.numpy() for k, v in params.items()}
    return {k: torch.tensor(v) for k, v in func(state, params).items()}


platings = [None, 1, 3]


@pytest.mark.parametrize("func", ab_xy_dynfuncs)
@pytest.mark.parametrize("xplatesize", platings)
@pytest.mark.parametrize("yplatesize", platings)
@pytest.mark.parametrize("aplatesize", platings)
@pytest.mark.parametrize("bplatesize", platings)
def test_prebroadcast(
    func,
    xplatesize,
    yplatesize,
    aplatesize,
    bplatesize,
):

    initial_state, atemp_params = ab_xy_prior(
        xplatesize, yplatesize, aplatesize, bplatesize
    )

    pre_broadcasted_initial_state = pre_broadcast_initial_state(
        func, initial_state, atemp_params=atemp_params
    )

    non_pre_broadcasted_return = exec_numpy_from_torch(
        func, initial_state, atemp_params
    )
    pre_broadcasted_return = exec_numpy_from_torch(
        func, pre_broadcasted_initial_state, atemp_params
    )

    for k in initial_state.keys():
        assert torch.allclose(non_pre_broadcasted_return[k], pre_broadcasted_return[k])
