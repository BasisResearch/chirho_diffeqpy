import logging
from importlib import import_module

import pyro
import pytest
import torch
from chirho.dynamical.internals.solver import check_dynamics
from chirho.dynamical.ops import State, simulate

from chirho_diffeqpy import ATempParams, DiffEqPy

pyro.settings.set(module_local_params=True)

logger = logging.getLogger(__name__)

# Global variables for tests
init_state = dict(S=torch.tensor(1.0))
start_time = torch.tensor(0.0)
end_time = torch.tensor(4.0)


def valid_diff(state: State, atemp_params: ATempParams) -> State:
    return state


def invalid_diff_bc_sample(state: State, atemp_params: ATempParams) -> State:
    pyro.sample("x", pyro.distributions.Normal(0.0, 1.0))
    return dict(S=(state["S"]))


def invalid_diff_bc_torch(state: State, atemp_params: ATempParams) -> State:
    return dict(S=torch.tensor(1.0))


@pytest.mark.parametrize(
    "lang_interop_backend", ["chirho_diffeqpy.lang_interop.julianumpy"]
)
def test_validate_dynamics_diffeqpy_sample(lang_interop_backend):
    solver = DiffEqPy

    # This loads the conversion operation overloads for a particular backend.
    # FIXME hk0jd16g will these overloads bleed into other tests?
    import_module(lang_interop_backend)

    with solver():
        check_dynamics(
            valid_diff,
            init_state,
            start_time,
            end_time,
            atemp_params=dict(),
        )

    with pytest.raises(ValueError) as e:
        with solver():
            check_dynamics(
                invalid_diff_bc_sample,
                init_state,
                start_time,
                end_time,
                atemp_params=dict(),
            )
    assert "does not allow `pyro.sample` calls" in str(e.value)


@pytest.mark.parametrize(
    "lang_interop_backend", ["chirho_diffeqpy.lang_interop.julianumpy"]
)
def test_validate_dynamics_setting_diffeqpy(lang_interop_backend):
    solver = DiffEqPy

    # This loads the conversion operation overloads for a particular backend.
    # FIXME hk0jd16g will these overloads bleed into other tests?
    import_module(lang_interop_backend)

    with pyro.settings.context(validate_dynamics=False):
        with solver(), solver():
            simulate(
                invalid_diff_bc_sample,
                init_state,
                start_time,
                end_time,
                atemp_params=dict(),
            )

    with pyro.settings.context(validate_dynamics=True):
        with pytest.raises(ValueError) as e:
            with solver(), solver():
                simulate(
                    invalid_diff_bc_sample,
                    init_state,
                    start_time,
                    end_time,
                    atemp_params=dict(),
                )
        assert "does not allow `pyro.sample` calls" in str(e.value)


@pytest.mark.parametrize(
    "lang_interop_backend", ["chirho_diffeqpy.lang_interop.julianumpy"]
)
def test_validate_dynamics_diffeqpy_torch(lang_interop_backend):
    solver = DiffEqPy

    # This loads the conversion operation overloads for a particular backend.
    # FIXME hk0jd16g will these overloads bleed into other tests?
    import_module(lang_interop_backend)

    with solver():
        check_dynamics(
            valid_diff,
            init_state,
            start_time,
            end_time,
            atemp_params=dict(),
        )

    with pytest.raises(NotImplementedError) as e:
        with solver():
            check_dynamics(
                invalid_diff_bc_torch,
                init_state,
                start_time,
                end_time,
                atemp_params=dict(),
            )
    assert "defined to operate on and return numpy arrays" in str(e.value)
