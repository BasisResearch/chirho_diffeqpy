from functools import singledispatch
from .fixtures import (
    MockClosureUnifiedFixtureDynamics,
)
from .fixtures_imported_from_chirho import (
    SIRObservationMixin,
    SIRReparamObservationMixin,
    UnifiedFixtureDynamics,
    bayes_sir_model,
    sir_param_prior,
    UnifiedFixtureDynamicsReparam,
    RandBetaUnifiedFixtureDynamics
)
from chirho.dynamical.handlers.solver import Solver
from .reparametrization import reparametrize_argument_by_value, reparametrize_argument_by_type
from chirho.dynamical.handlers.solver import TorchDiffEq
from .mock_closure import  DiffEqPyMockClosureCapable


# <Solver>
# Given any instance of a Solver subclass....
@reparametrize_argument_by_type.register(Solver)
def _(*args, **kwargs):
    # ...return instance of MockClosure capable DiffEqPy solver.
    return DiffEqPyMockClosureCapable()


# Given the TorchDiffEq class itself...
@reparametrize_argument_by_value.register(TorchDiffEq)
def _(*args, **kwargs):
    # ...still return an instance. See DiffEqPyMockClosureCapable.__call__ for the rationale.
    return DiffEqPyMockClosureCapable()
# </Solver>


# <Dynamics>
# Given any instance of UnifiedFixtureDynamics...
@reparametrize_argument_by_type.register(UnifiedFixtureDynamics)
def generate_diffeqpy_bayes_sir_model(dispatch_arg, *args, **kwargs):
    # ...return instance of a MockDynamicsUnifiedFixtureDynamics with the same parameter tensors.
    return MockClosureUnifiedFixtureDynamics(
        beta=dispatch_arg.beta,
        gamma=dispatch_arg.gamma
    )


# Given the bayes_sir_model function itself...
@reparametrize_argument_by_value.register(bayes_sir_model)
def _(dispatch_arg, *args, **kwargs):
    # ...return a function that converts the returned model to a mock closure.
    return lambda: generate_diffeqpy_bayes_sir_model(dispatch_arg())


# Given the specific fixture dynamics subclass itself...
@reparametrize_argument_by_value.register(RandBetaUnifiedFixtureDynamics)
def _(dispatch_arg, *args, **kwargs):
    assert dispatch_arg is RandBetaUnifiedFixtureDynamics
    # ...return a class that overrides its forward (by preceding in MRO) but allows it to specify
    #  its special getter for the beta parameter.
    class MockClosureRandBetaUnifiedFixtureDynamics(MockClosureUnifiedFixtureDynamics, RandBetaUnifiedFixtureDynamics):
        pass

    return MockClosureRandBetaUnifiedFixtureDynamics

# </Dynamics>
