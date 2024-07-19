from typing import TypeVar, Generic, Optional
from chirho_diffeqpy import Dynamics, ATempParams, DiffEqPy
from chirho.dynamical.ops import simulate, State
from pyro.poutine.messenger import Messenger
from pyro.poutine.messenger import block_messengers
from deepdiff import DeepDiff

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

    def __call__(self, fn=None, **kwargs):
        """
        A hack for tests that returns self if this entity is being called in a context where it was reparametrized
        for its type. This lets us pass one solver instance into a test that expects Solver types, so that compilation
        is performed once wrt the entire chirho test.
        Note that the nuances of the compilation and caching are tested internally, and not via the chirho tests.
        """
        # This is likely being called as a handler, and needs to
        if fn is not None:
            if callable(fn):
                return super().__call__(fn)
            else:
                raise ValueError("DiffEqPyMockClosureCapable should be called with a callable, "
                                 " or with no arguments at all.")

        kwarg_diff = DeepDiff(self.solve_kwargs, kwargs)
        if len(kwarg_diff):
            raise ValueError("An instance of DiffEqPyMockClosureCapable was called with different kwargs than it was "
                             f"initialized with. The diff was: {kwarg_diff}")

        return self
