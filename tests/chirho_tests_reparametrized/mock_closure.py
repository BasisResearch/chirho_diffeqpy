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


class DiffEqPyMockClosureCapable(DiffEqPy):
    """
    This enables closure-style dispatch of simulate, which allows us to operate with the same dynamics/parameter
    coupling that the chirho tests use by defining dynamics as torch modules.
    """

    # noinspection PyMethodMayBeStatic,PyFinal
    def _pyro_simulate(self, msg: dict) -> None:
        dynamics = msg["args"][0]

        if isinstance(dynamics, MockDynamicsClosureAbstract):
            if "atemp_params" in msg["kwargs"]:
                raise ValueError("MockDynamicsClosureAbstract instances should not have atemp_params passed in "
                                 "to simulate, but rather exposed via the overriden atemp_params property.")
            msg["kwargs"]["atemp_params"] = dynamics.atemp_params
            return super()._pyro_simulate(msg)
