from functools import singledispatch
from .fixtures_imported_from_chirho import UnifiedFixtureDynamics
from chirho.dynamical.handlers.solver import Solver
from chirho_diffeqpy import DiffEqPy
from .fixtures import sir_dynamics, sample_sir_params
from .per_test_reparametrization import per_test_reparametrization


@singledispatch
def reparametrize_argument(primary_arg, *secondary_args, test_id: str = None, arg_names: str = None):
    # TODO the test specific rules should probably take precedent?
    # If no general reparametrization applies, then we need to dispatch to test-specific reparametrizations.
    if test_id in per_test_reparametrization:
        return per_test_reparametrization[test_id](primary_arg, *secondary_args, arg_names=arg_names)
    else:
        raise NotImplementedError("This type does not yet have a reparametrization implemented, and no test-specific "
                                  f"reparametrization was found for test {test_id}. Please add a general conversation"
                                  f" rule or test-specific conversion for {primary_arg}, {secondary_args}.")


# Note that singledispatch only gets us part of the way here, so we'll break up the dispatch mechanisms where needed
#  into messier sub-dispatching.


#######################################################################################################################
# Reparam of instantiated solver objects
#######################################################################################################################
@reparametrize_argument.register
def _(primary_arg: Solver, test_id: str = None, arg_names: str = None):
    return DiffEqPy()


#######################################################################################################################
# Reparam of types by their value.
#######################################################################################################################
@reparametrize_argument.register
def _reparametrize_type(primary_arg: type, test_id: str = None, arg_names: str = None):
    # singledispatch, afaik, can't dispatch on values of type Type (also, just on values in general).
    if issubclass(primary_arg, Solver):
        return DiffEqPy
    else:
        raise NotImplementedError("This type does not yet have a reparametrization implemented.")


#######################################################################################################################
# Reparam of unpacked tuples according to their first argument.
#######################################################################################################################
@reparametrize_argument.register
def _(primary_arg: UnifiedFixtureDynamics, simulation_kwargs: dict, test_id: str = None, arg_names: str = None):
    return sir_dynamics, dict(atemp_params=sample_sir_params())


@reparametrize_argument.register
def _unpack_tuple(primary_arg: tuple, **kwargs):
    # Tuples will be forwarded on to be reparametrized as a group, based on their first argument.
    return reparametrize_argument(*primary_arg, **kwargs)
