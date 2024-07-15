from functools import singledispatch
from .fixtures_from_chirho import UnifiedFixtureDynamics
from chirho.dynamical.handlers.solver import Solver
from chirho_diffeqpy import DiffEqPy
from .fixtures import sir_dynamics, sample_sir_params


@singledispatch
def reparametrize_argument(primary_arg, *secondary_args):
    raise NotImplementedError("This type does not yet have a reparametrization implemented.")


# Note that singledispatch only gets us part of the way here, so we'll break up the dispatch mechanisms where needed
#  into messier sub-dispatching.


#######################################################################################################################
# Reparam of instantiated solver objects
#######################################################################################################################
@reparametrize_argument.register
def _(primary_arg: Solver):
    return DiffEqPy()


#######################################################################################################################
# Reparam of types by their value.
#######################################################################################################################
@reparametrize_argument.register
def _reparametrize_type(primary_arg: type):
    # singledispatch, afaik, can't dispatch on values of type Type (also, just on values in general).
    if issubclass(primary_arg, Solver):
        return DiffEqPy
    else:
        raise NotImplementedError("This type does not yet have a reparametrization implemented.")


#######################################################################################################################
# Reparam of unpacked tuples according to their first argument.
#######################################################################################################################
@reparametrize_argument.register
def _(primary_arg: UnifiedFixtureDynamics, simulation_kwargs: dict):
    return sir_dynamics, sample_sir_params()


@reparametrize_argument.register
def _unpack_tuple(primary_arg: tuple):
    # Tuples will be forwarded on to be reparametrized as a group, based on their first argument.
    return reparametrize_argument(*primary_arg)
