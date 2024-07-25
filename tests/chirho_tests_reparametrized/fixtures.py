import numpy as np
import torch
from .fixtures_imported_from_chirho import (
    UnifiedFixtureDynamicsBase,
    SIRObservationMixin,
    SIRReparamObservationMixin,
    get_state_reached_event_f
)
from .mock_closure import MockDynamicsClosureAbstract
from chirho_diffeqpy.handlers import State, ATempParams
from chirho_diffeqpy.internals import to_numpy


def isalambda(f):
    return callable(f) and f.__name__ == "<lambda>"


# Exactly as written in the original chirho fixture, but uses np.sin instead of torch.sin....
# TODO md7291jdmd dispatch sin on type in the original chirho code so we can use that one by just defining a sin
#  dispatch on numpy arrays?
def pure_sir_dynamics(state, atemp_params):
    beta = atemp_params["beta"]
    gamma = atemp_params["gamma"]

    dX = dict()

    beta = beta * (
            1.0 + 0.1 * np.sin(0.1 * state["t"])
    )  # beta oscilates slowly in time.

    dX["S"] = -beta * state["S"] * state["I"]  # noqa
    dX["I"] = beta * state["S"] * state["I"] - gamma * state["I"]  # noqa
    dX["R"] = gamma * state["I"]  # noqa

    return dX


class MockClosureUnifiedFixtureDynamicsBase(MockDynamicsClosureAbstract, UnifiedFixtureDynamicsBase):

    def __init__(self, beta=None, gamma=None):
        super().__init__(
            pure_dynamics=pure_sir_dynamics,
            beta=beta,
            gamma=gamma
        )

    @property
    def atemp_params(self):
        return dict(beta=self.beta, gamma=self.gamma)


class MockClosureUnifiedFixtureDynamics(MockClosureUnifiedFixtureDynamicsBase, SIRObservationMixin):
    pass


class MockClosureUnifiedFixtureDynamicsReparam(MockClosureUnifiedFixtureDynamicsBase, SIRReparamObservationMixin):
    pass


class MockClosureDynamicsDirectPass(MockDynamicsClosureAbstract):

    def __init__(self, dynamics, atemp_params):
        super().__init__(pure_dynamics=dynamics)
        self._atemp_params = atemp_params

    @property
    def atemp_params(self):
        return self._atemp_params


def build_state_reached_pure_event_fn(target_state: State[torch.tensor], event_dim: int = 0):
    target_state_np = to_numpy(target_state)
    impure_event_f = get_state_reached_event_f(target_state_np, event_dim)

    def pure_event_f(t: float, state: State[np.ndarray], atemp_params: ATempParams[np.ndarray]):
        return impure_event_f(t, state)

    return pure_event_f
