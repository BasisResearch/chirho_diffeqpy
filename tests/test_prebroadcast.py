import juliacall  # Must precede even indirect torch imports to preventshape segfault.
import pytest
import torch
from chirho_diffeqpy.internals import pre_broadcast
from contextlib import nullcontext
import pyro
import pyro.distributions as dist
from .fixtures import ab_xy_dynfuncs


def prior(
    xplatesize,
    yplatesize,
    aplatesize,
    bplatesize,
):
    xplate = pyro.plate("xplate", xplatesize, dim=-4) if xplatesize is not None else nullcontext()
    yplate = pyro.plate("yplate", yplatesize, dim=-3) if yplatesize is not None else nullcontext()
    aplate = pyro.plate("aplate", aplatesize, dim=-2) if aplatesize is not None else nullcontext()
    bplate = pyro.plate("bplate", bplatesize, dim=-1) if bplatesize is not None else nullcontext()

    with xplate:
        x = pyro.sample("x", dist.Normal(0, 1))
        with yplate:
            y = pyro.sample("y", dist.Normal(0, 1))

    with aplate:
        a = pyro.sample("a", dist.Normal(0, 1))
        with bplate:
            b = pyro.sample("b", dist.Normal(0, 1))

    return dict(x=x, y=y), dict(a=a, b=b)


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

    initial_state, atemp_params = prior(xplatesize, yplatesize, aplatesize, bplatesize)

    pre_broadcasted_initial_state = pre_broadcast(func, initial_state, atemp_params)

    non_pre_broadcasted_return = exec_numpy_from_torch(func, initial_state, atemp_params)
    pre_broadcasted_return = exec_numpy_from_torch(func, pre_broadcasted_initial_state, atemp_params)

    for k in initial_state.keys():
        assert torch.allclose(non_pre_broadcasted_return[k], pre_broadcasted_return[k])
