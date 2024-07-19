from functools import singledispatch
from .fixtures import (
    pure_sir_dynamics
)
from .fixtures_imported_from_chirho import (
    SIRObservationMixin,
    SIRReparamObservationMixin,
    UnifiedFixtureDynamics,
    bayes_sir_model,
    sir_param_prior,
    UnifiedFixtureDynamicsReparam
)
from chirho.dynamical.handlers.solver import Solver
from .reparametrization import reparametrize_argument_by_value, reparametrize_argument_by_type
from chirho.dynamical.handlers.solver import TorchDiffEq
from .mock_closure import MockDynamicsClosureDirectPass, DiffEqPyMockClosureCapable


# <Solver>
# Given any instance of a Solver subclass....
@reparametrize_argument_by_type.register(Solver)
def _(*args, **kwargs):
    # ...return instance of MockClosure capable DiffEqPy solver.
    return DiffEqPyMockClosureCapable()


# Given the TorchDiffEq class itself...
@reparametrize_argument_by_value.register(TorchDiffEq)
def _(*args, **kwargs):
    # ...return the MockClosure capable DiffEqPy solver class itself.
    return DiffEqPyMockClosureCapable
# </Solver>


# <Dynamics>
# Given any instance of UnifiedFixtureDynamics...
@reparametrize_argument_by_type.register(UnifiedFixtureDynamics)
def generate_diffeqpy_bayes_sir_model(*args, **kwargs):
    # ...return instance of a MockDynamicsClosureDirectPass, taking the dynamics and exposing the atemp_params
    #  to DiffEqPyMockClosureCapable for passing directly into the simulate call via a kwarg, as is expected by
    #  the chirho_diffeqpy api, but handled here in a way that will work with the way the chirho tests are written.
    beta, gamma = sir_param_prior()
    return MockDynamicsClosureDirectPass(
        dynamics=pure_sir_dynamics,
        atemp_params=dict(beta=beta, gamma=gamma)
    )


# Given the bayes_sir_model function itself...
@reparametrize_argument_by_value.register(bayes_sir_model)
def _(*args, **kwargs):
    # ...return a function that builds the model, just like bayes_sir_model.
    return generate_diffeqpy_bayes_sir_model
# </Dynamics>


print("for breakpoint")