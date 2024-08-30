from chirho.dynamical.ops import State
from deepdiff import DeepDiff
from torch import Tensor as Tnsr

from chirho_diffeqpy import (
    ATempParams,
    DiffEqPyMockClosureCapable,
    MockDynamicsClosureMixin,
    PureDynamics,
)


class MockDynamicsClosurePassPure(MockDynamicsClosureMixin):
    """
    Used in tandem with DiffEqPyMockClosureCapable to enable closure-style dispatch of simulate.
    """

    def __init__(self, pure_dynamics: PureDynamics[Tnsr], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pure_dynamics = pure_dynamics

    def forward(
        self, state: State[Tnsr], atemp_params: ATempParams[Tnsr]
    ) -> State[Tnsr]:
        return self.pure_dynamics(state, atemp_params)


class DiffEqPyMockClosureCapableSingleCompile(DiffEqPyMockClosureCapable):

    def __call__(self, fn=None, **kwargs):
        """
        A hack for tests that returns self if this entity is being called in a context where it was reparametrized
         for its type. This lets us pass one solver INSTANCE into a test that expects Solver TYPES, so that compilation
         is performed in one scope wrt the entire chirho test.
        Note that the nuances of the compilation and caching are tested internally in this repo,
         and not via the chirho tests.
        """
        # This is likely being called as a handler, and needs to
        if fn is not None:
            if callable(fn):
                return super().__call__(fn)
            else:
                raise ValueError(
                    "DiffEqPyMockClosureCapable should be called with a callable, "
                    " or with no arguments at all."
                )

        kwarg_diff = DeepDiff(self.solve_kwargs, {**self.DEFAULT_KWARGS, **kwargs})
        if len(kwarg_diff):
            raise ValueError(
                "An instance of DiffEqPyMockClosureCapable was called with different kwargs than it was "
                f"initialized with. The diff was: {kwarg_diff}"
            )

        return self
