from typing import TypeVar, Generic, Optional
from chirho_diffeqpy import Dynamics, ATempParams, DiffEqPy
from chirho.dynamical.ops import simulate, State
from pyro.poutine.messenger import Messenger
from pyro.poutine.messenger import block_messengers

T = TypeVar("T")


class MockDynamicsClosureAbstract(Generic[T]):
    def __init__(self, dynamics: Dynamics[T]):
        self.dynamics = dynamics

    @property
    def atemp_params(self):
        raise NotImplementedError()

    def __call__(self, state: State[T], atemp_params: ATempParams[T]) -> State[T]:
        return self.dynamics(state, atemp_params)


class MockDynamicsClosureDirectPass(MockDynamicsClosureAbstract[T]):
    def __init__(self, dynamics: Dynamics[T], atemp_params: ATempParams[T]):
        super().__init__(dynamics)
        self._atemp_params = atemp_params

    @property
    def atemp_params(self):
        return self._atemp_params


class _MockDynamicsClosureHandler(Messenger):

    # noinspection PyMethodMayBeStatic
    def _pyro_simulate(self, msg: dict) -> None:

        dynamics = msg["args"]

        if isinstance(dynamics, MockDynamicsClosureAbstract):
            with block_messengers(lambda m: isinstance(m, _MockDynamicsClosureHandler)):
                msg["value"] = simulate(dynamics, msg["args"][1:], **msg["kwargs"], atemp_params=dynamics.atemp_params)
                msg["done"] = True
        # Otherwise, this will propagate to the default implementation.


class DiffEqPyMockClosureCapable(DiffEqPy, _MockDynamicsClosureHandler):
    """
    This enables closure-style dispatch of simulate, which allows us to operate with the same dynamics/parameter
    coupling that the chirho tests use by defining dynamics as torch modules.
    """
    pass
