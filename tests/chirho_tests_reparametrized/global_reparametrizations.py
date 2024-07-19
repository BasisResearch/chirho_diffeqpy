from functools import singledispatch
from .fixtures_imported_from_chirho import (
    pure_sir_dynamics,
    SIRObservationMixin,
    SIRReparamObservationMixin,
    UnifiedFixtureDynamics,
    bayes_sir_model,
    sir_param_prior,
    UnifiedFixtureDynamicsReparam
)
from chirho.dynamical.handlers.solver import Solver
from chirho_diffeqpy import DiffEqPy
from .ops import reparametrize_argument_by_value, reparametrize_argument_by_type


@reparametrize_argument_by_type.register
def _(dispatch_arg: Solver, *args, **kwargs):
    return DiffEqPy()


@reparametrize_argument_by_value.register
def _(dispatch_arg: