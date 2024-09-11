import torch
from chirho.dynamical.handlers.solver import Solver, TorchDiffEq

from .fixtures import (
    MockClosureDynamicsDirectPass,
    MockClosureUnifiedFixtureDynamics,
    MockClosureUnifiedFixtureDynamicsReparam,
    build_state_reached_pure_event_fn,
)
from .fixtures_imported_from_chirho import (  # noqa: F401
    RandBetaUnifiedFixtureDynamics,
    SIRObservationMixin,
    SIRReparamObservationMixin,
    UnifiedFixtureDynamics,
    UnifiedFixtureDynamicsReparam,
    bayes_sir_model,
    build_event_fn_zero_after_tt,
    get_state_reached_event_f,
    model_with_param_in_state,
    sir_param_prior,
)
from .mock_closure import DiffEqPyMockClosureCapable
from .reparametrization import (
    reparametrize_argument_by_type,
    reparametrize_argument_by_value,
)


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
        beta=dispatch_arg.beta, gamma=dispatch_arg.gamma
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
    class MockClosureRandBetaUnifiedFixtureDynamics(
        MockClosureUnifiedFixtureDynamics, RandBetaUnifiedFixtureDynamics
    ):
        pass

    return MockClosureRandBetaUnifiedFixtureDynamics


@reparametrize_argument_by_value.register(model_with_param_in_state)
def _(*args, **kwargs):

    # Annoyingly, dX has to be a symbolic function of state, so for constants we have to do hacky stuff.
    # Multiplying or dividing by the state itself preserves shape information.
    def model_with_param_in_state_np(X):
        dX = dict()
        dX["x"] = (X["x"] + 1.0) / (X["x"] + 1.0)  # a constant, 1.
        dX["z"] = X["dz"]
        dX["dz"] = X["dz"] * 0.0  # also a constant, this gets set by interventions.
        dX["param"] = (
            X["param"] * 0.0
        )  # this is a constant event function parameter, so no change.
        return dX

    return MockClosureDynamicsDirectPass(
        dynamics=lambda state, atemp_params: model_with_param_in_state_np(state),
        atemp_params=dict(),
    )


# Given any instance of UnifiedFixtureDynamicsReparam
@reparametrize_argument_by_type.register(UnifiedFixtureDynamicsReparam)
def _(dispatch_arg, *args, **kwargs):
    return MockClosureUnifiedFixtureDynamicsReparam(
        beta=dispatch_arg.beta, gamma=dispatch_arg.gamma
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
