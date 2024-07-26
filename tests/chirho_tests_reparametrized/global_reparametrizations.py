from functools import singledispatch
from .fixtures import (
    MockClosureUnifiedFixtureDynamics,
    build_state_reached_pure_event_fn,
    MockClosureDynamicsDirectPass
)
from .fixtures_imported_from_chirho import (
    SIRObservationMixin,
    SIRReparamObservationMixin,
    UnifiedFixtureDynamics,
    bayes_sir_model,
    sir_param_prior,
    UnifiedFixtureDynamicsReparam,
    RandBetaUnifiedFixtureDynamics,
    build_event_fn_zero_after_tt,
    get_state_reached_event_f,
    model_with_param_in_state
)
from chirho.dynamical.handlers.solver import Solver
from .reparametrization import reparametrize_argument_by_value, reparametrize_argument_by_type
from chirho.dynamical.handlers.solver import TorchDiffEq
from .mock_closure import  DiffEqPyMockClosureCapable
import torch
import numpy as np


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


@reparametrize_argument_by_value.register(model_with_param_in_state)
def _(*args, **kwargs):

    # The same as the original chirho fixture but with scalars (not torch tensors) as constants.
    def model_with_param_in_state_np(X):
        dX = dict()
        dX["x"] = np.array(1.0)
        dX["z"] = X["dz"]
        dX["dz"] = np.array(0.0)  # also a constant, this gets set by interventions.
        dX["param"] = np.array(0.0)  # this is a constant event function parameter, so no change.
        return dX

    return MockClosureDynamicsDirectPass(
        dynamics=lambda state, atemp_params: model_with_param_in_state_np(state),
        atemp_params=dict()
    )
# </Dynamics>


# <Event Functions>
@reparametrize_argument_by_value.register(build_event_fn_zero_after_tt)
def _(dispatch_arg, *args, **kwargs):
    assert dispatch_arg is build_event_fn_zero_after_tt

    def build_event_fn_zero_after_tt_np(tt: torch.Tensor):
        tt = tt.item()

        def zero_after_tt(t, state, atemp_params):
            return tt - t

        return zero_after_tt

    return build_event_fn_zero_after_tt_np


@reparametrize_argument_by_value.register(get_state_reached_event_f)
def _(dispatch_arg, *args, **kwargs):
    assert dispatch_arg is get_state_reached_event_f
    return build_state_reached_pure_event_fn

# </Event Functions>
