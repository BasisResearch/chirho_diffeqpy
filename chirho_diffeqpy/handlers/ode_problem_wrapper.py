from diffeqpy import de
from ..internals import (
    ATempParams,
    State,
    _flatten_mapping,
    get_mapping_shape,
    pre_broadcast_initial_state
)
import numpy as np


class ODEProblemWrapper:
    def __init__(self, problem: de.ODEProblem):
        self.problem = problem

    def __call__(self, state: State[np.ndarray], atemp_params: ATempParams[np.ndarray]) -> State[np.ndarray]:
        raise TypeError(
            "Cannot call wrapped, pre-compiled ODEProblems directly, but they can still be passed to simulate as "
            " a dynamics function."
        )

    def check_shapes(self, state: State[np.ndarray], atemp_params: ATempParams[np.ndarray]):
        flat_state = _flatten_mapping(state)
        flat_atemp_params = _flatten_mapping(atemp_params)

        if self.problem.u0.shape != flat_state.shape:
            raise ValueError(
                f"State has shape:\n{get_mapping_shape(state)}\n"
                f"But this cannot be flattened down to the shape "
                f"of the ODEProblem's state {self.problem.u0.shape}."
                f" Instead, it flattens to {flat_state.shape}."
            )

        if self.problem.p.shape != flat_atemp_params.shape:
            raise ValueError(
                f"Parameters have shape:\n{get_mapping_shape(atemp_params)}\n"
                f"But this cannot be flattened down to the shape "
                f"of the ODEProblem's parameters {self.problem.p.shape}."
                f" Instead, it flattens to {flat_atemp_params.shape}."
            )

        return True


@pre_broadcast_initial_state.register
def _(dynamics: ODEProblemWrapper, initial_state: State, atemp_params: ATempParams) -> State:
    # We can't actually execute the julia dynamics function here to pre-broadcast, so the user will have to manage
    #  this themselves for now.
    # FIXME WIP this might actually "just work" b/c the compilation will be wrt symbolic arrays presumably? So
    #  so broadcasting might work? Eh maybe not though, it seems like the ODEProblem is locked into a particular
    #  state shape.
    try:
        dynamics.check_shapes(initial_state, atemp_params)
    except ValueError as e:
        raise ValueError(
            f"ODEProblem shapes do not match passed state and parameters. If the shape should match,"
            f" you may need to manually pre-broadcast the state over the parameters, or provide an ODEProblem"
            f" that supports the relevant shape."
        )
    return initial_state
