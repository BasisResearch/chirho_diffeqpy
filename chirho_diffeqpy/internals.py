from typing import Callable, List, Optional, Tuple, Mapping, TypeVar

import pyro
import torch
from diffeqpy import de

from chirho.dynamical.internals._utils import _squeeze_time_dim, _var_order
from chirho.dynamical.internals.solver import Interruption, simulate_point
from chirho.dynamical.ops import State
from chirho.indexed.ops import IndexSet, gather, get_index_plates
from torch import Tensor as Tnsr
from juliatorch import JuliaFunction
import juliacall
import numpy as np
from typing import Union
from functools import singledispatch
from chirho_diffeqpy.lang_interop import callable_from_julia
from juliacall import Main as jl

T = TypeVar("T")
ATempParams = Mapping[str, T]
Dynamics = Callable[[State[T], ATempParams[T]], State[T]]

_ShapeType = Tuple[int, ...]
_NamedShape = Tuple[str, _ShapeType]
MappingShape = Tuple[_NamedShape, ...]


def get_mapping_shape(mapping: Mapping[str, Tnsr]) -> MappingShape:
    var_order = get_var_order(mapping)
    return tuple((_var, mapping[_var].shape) for _var in var_order)


def pre_broadcast(
        dynamics: Dynamics[np.ndarray],
        initial_state_torch: State[Tnsr],
        atemp_params_torch: ATempParams[Tnsr]
):
    """
    Prebroadcasts an initial state across the parameters so that compiled dynamics can be made with respect to the
    same input and output state shapes.
    :param dynamics:
    :param initial_state_torch:
    :param atemp_params_torch:
    :return:
    """
    # Detach and convert everything to numpy arrays.
    initial_state_np = {k: v.detach().numpy() for k, v in initial_state_torch.items()}
    atemp_params_np = {k: v.detach().numpy() for k, v in atemp_params_torch.items()}

    if "t" not in initial_state_np:
        initial_state_np["t"] = 0.0

    # Evaluate the dynamics to get the output shape — this corresponds to the broadcast-induced state shape that
    #  we actually need to be solving for. Note: the broadcast shape will typically be induced by plating on the params.
    output = dynamics(initial_state_np, atemp_params_np)

    # Now, we just want to broadcast the initial state to the shape of the output. To do so, we can just add zeros_like
    #  output to the initial_state.
    broadcasted_initial_state = {k: initial_state_torch[k] + np.zeros_like(torch.tensor(output[k]))
                                 for k, v in initial_state_torch.items()}

    return broadcasted_initial_state


def diffeqdotjl_compile_problem(
    dynamics: Dynamics[np.ndarray],
    initial_state: State[Tnsr],
    start_time: Tnsr,
    end_time: Tnsr,
    atemp_params: ATempParams[Tnsr],
    **kwargs
) -> de.ODEProblem:

    # TODO take general compilation kwargs here too.

    require_float64(initial_state)
    require_float64(atemp_params)

    # See the note below for why this must be a pure function and cannot use the values in initial_atemp_params directly.
    # callable_from_julia(out_as_first_arg) just specifies that it expects julia to call it with the preallocated
    #  output array as the first argument (as opposed to listing out as a keyword, which is the default expectation).
    # diffeqpy, when compiling the dynamics function, passes the output array as the first argument.
    @callable_from_julia(out_as_first_arg=True)
    def ode_f(flat_state, flat_inner_atemp_params, t):
        # Unflatten the state u according to the state variables stored in initial dstate.
        state: State = _unflatten_mapping(
            flat_state,  # WIP NOTE: this used to be JuliaThingWrapper.wrap_array(flat_state)
            initial_state
        )

        # Note that initial_atemp_params will be a dictionary of shaped torch tensors, while flat_atemp_params will be
        # a vector of julia symbolics involved in jit copmilation.
        inner_atemp_params: ATempParams = _unflatten_mapping(
            flattened_mapping=flat_inner_atemp_params,
            shaped_mapping_for_reference=atemp_params
        ) if len(flat_inner_atemp_params) > 0 else dict()

        if "t" in state:
            raise ValueError("Cannot have a state variable named 't' as it is reserved for the time variable.")
        state = dict(**state, t=t)

        dstate = dynamics(state, inner_atemp_params)

        flat_dstate = _flatten_mapping(dstate)

        return flat_dstate

    # Pre-broadcast the initial state so that we compile the dynamics function with the proper state-space dimensions
    #  (as induced, typically, by the dynamics broadcasting the parameters over an unexpanded initial state).
    initial_state = pre_broadcast(dynamics, initial_state, atemp_params)

    # Flatten the initial state and parameters.
    flat_initial_state = _flatten_mapping(initial_state)
    flat_atemp_params = _flatten_mapping(atemp_params)

    # See juliatorch readme to motivate the FullSpecialize syntax.
    prob = de.seval("ODEProblem{true, SciMLBase.FullSpecialize}")(
        ode_f,
        # Note that we are only interested in compilation here, not differentiation! These values will strictly be
        #  used for their types and shapes so that the ODEProblem knows what to compile for.
        flat_initial_state.detach().numpy(),
        np.array([start_time, end_time], dtype=np.float64),
        # HACK sgfeh2 stuff breaks with a len zero array, and passing nothing or None incurs the same problem.
        flat_atemp_params.detach().numpy() if len(flat_atemp_params) else np.ones((1,), dtype=np.float64),
    )

    fast_prob = de.jit(prob)

    return fast_prob


# TODO g179du91 move to internal ops as other backends might also use this?
# See use in handlers/solver.DiffEqDotJL._pyro__lazily_compile_problem
@pyro.poutine.runtime.effectful(type="_lazily_compile_problem")
def _lazily_compile_problem(*args, **kwargs) -> de.ODEProblem:
    raise NotImplementedError()


def get_var_order(mapping: Mapping[str, Tnsr]) -> Tuple[str, ...]:
    return _var_order(frozenset(mapping.keys()))


# Single dispatch cat thing that handles both tensors and numpy arrays.
@singledispatch
def flat_cat(*vs):
    # Default implementation assumes we're dealing some underying julia thing that needs to be put back into an array.
    # This will re-route to the numpy implementation.

    # If atleast_1d receives a single argument, it will return a single array, rather than a tuple of arrays.
    vs = np.atleast_1d(*vs)
    return flat_cat_numpy(*(vs if isinstance(vs, (list, tuple)) else (vs,)))


@flat_cat.register
def flat_cat_torch(*vs: Tnsr):
    return torch.cat([v.ravel() for v in vs])


@flat_cat.register
def flat_cat_numpy(*vs: np.ndarray):
    return np.concatenate([v.ravel() for v in vs])


# TODO ofwr1948 replace with generic pytree flatten/unflatten?
def _flatten_mapping(mapping: Mapping[str, Union[Tnsr, np.ndarray]]) -> Union[Tnsr, np.ndarray]:
    if len(mapping) == 0:
        # TODO do17bdy1t address type specificity
        return torch.tensor([], dtype=torch.float64)
    var_order = get_var_order(mapping)
    return flat_cat(*[mapping[v] for v in var_order])


# TODO ofwr1948 replace with generic pytree flatten/unflatten?
def _unflatten_mapping(
        flattened_mapping: Tnsr,
        shaped_mapping_for_reference: Mapping[str, Union[Tnsr, juliacall.VectorValue]],
        to_traj: bool = False
) -> Mapping[str, Union[Tnsr, np.ndarray]]:

    var_order = get_var_order(shaped_mapping_for_reference)
    mapping_to_return = dict()
    for v in var_order:
        shaped = shaped_mapping_for_reference[v]
        shape = shaped.shape
        if to_traj:
            # If this is a trajectory of states, the dimension following the original shape's state will be time,
            #  and because we know the rest of the shape we can auto-detect its size with -1.
            shape += (-1,)

        sv = flattened_mapping[:shaped.numel()].reshape(shape)

        mapping_to_return[v] = sv

        # Slice so that only the remaining elements are left.
        flattened_mapping = flattened_mapping[shaped.numel():]

    return mapping_to_return


# TODO ofwr1948 replace with generic pytree traversal.
def require_float64(mapping: Mapping[str, Tnsr]):
    # Forward diff through diffeqpy currently requires float64. # TODO do17bdy1t update when this is fixed.
    for k, v in mapping.items():
        if v.dtype is not torch.float64:
            raise ValueError(f"State or parameter variable {k} has dtype {v.dtype}, but must be float64.")


def _diffeqdotjl_ode_simulate_inner(
    dynamics: Dynamics[np.ndarray],
    initial_state: State[Tnsr],
    timespan: Tnsr,
    atemp_params: ATempParams[Tnsr],
    **kwargs
) -> State[torch.tensor]:

    if not isinstance(atemp_params, dict):
        raise ValueError(f"atemp_params must be a dictionary, "
                         f"but got type {type(atemp_params)} and value {atemp_params}")

    require_float64(initial_state)
    require_float64(atemp_params)

    compiled_prob = _lazily_compile_problem(
        dynamics,
        # Note: these inputs are only used on the first compilation so that the types, shapes etc. get compiled along
        # with the problem. Subsequently (in the inner_solve), these are ignored (even though they have the same as
        # exact values as the args passed into `remake` below). The outer_u0_t_p has to be passed into the
        # JuliaFunction.apply so that those values can be put into Dual numbers by juliatorch.
        initial_state,
        timespan[0],
        timespan[-1],
        atemp_params=atemp_params,
        **kwargs,
    )

    initial_state = pre_broadcast(dynamics, initial_state, atemp_params)

    # Flatten the initial state, timespan, and parameters into a single vector. This is required because
    #  juliatorch currently requires a single matrix or vector as input.
    flat_initial_state = _flatten_mapping(initial_state)
    flat_atemp_params = _flatten_mapping(atemp_params)
    # HACK sgfeh2 stuff breaks with a len zero array, and passing nothing or None incurs the same problem.
    flat_atemp_params = torch.ones((1,), dtype=torch.float64) if len(flat_atemp_params) == 0 else flat_atemp_params
    outer_u0_t_p = torch.cat([flat_initial_state, timespan, flat_atemp_params])

    def inner_solve(u0_t_p):

        # Unpack the concatenated initial state, timespan, and parameters.
        u0 = u0_t_p[:flat_initial_state.numel()]
        tspan = u0_t_p[flat_initial_state.numel():-flat_atemp_params.numel()]
        p = u0_t_p[-flat_atemp_params.numel():]

        # Remake the otherwise-immutable problem to use the new parameters.
        remade_compiled_prob = de.remake(compiled_prob, u0=u0, p=p, tspan=(tspan[0], tspan[-1]))

        sol = de.solve(remade_compiled_prob)

        # Interpolate the solution at the requested times.
        return sol(tspan)

    # Finally, execute the juliacall function
    flat_traj = JuliaFunction.apply(inner_solve, outer_u0_t_p)

    # Unflatten the trajectory.
    # return _unflatten_state(flat_traj, initial_state, to_traj=True)
    return _unflatten_mapping(
        flattened_mapping=flat_traj,
        shaped_mapping_for_reference=initial_state,
        to_traj=True
    )


def diffeqdotjl_simulate_trajectory(
    dynamics: Dynamics[np.ndarray],
    initial_state: State[Tnsr],
    timespan: Tnsr,
    **kwargs,
) -> State[Tnsr]:
    return _diffeqdotjl_ode_simulate_inner(dynamics, initial_state, timespan, **kwargs)


def diffeqdotjl_simulate_to_interruption(
    interruptions: List[Interruption],
    dynamics: Dynamics[np.ndarray],
    initial_state: State[Tnsr],
    start_time: Tnsr,
    end_time: Tnsr,
    atemp_params: ATempParams[Tnsr],
    **kwargs,
) -> Tuple[State[Tnsr], Tnsr, Optional[Interruption]]:

    # TODO TODO implement the actual retrieval of the next interruption (see torchdiffeq_simulate_to_interruption)

    from chirho.dynamical.handlers.interruption import StaticInterruption
    next_interruption = StaticInterruption(end_time)

    value = simulate_point(
        dynamics, initial_state, start_time, end_time, atemp_params=atemp_params, **kwargs
    )

    return value, end_time, next_interruption


def diffeqdotjl_simulate_point(
    dynamics: Dynamics[np.ndarray],
    initial_state: State[torch.Tensor],
    start_time: torch.Tensor,
    end_time: torch.Tensor,
    atemp_params: ATempParams[T],
    **kwargs,
) -> State[torch.Tensor]:

    timespan = torch.stack((start_time, end_time))
    trajectory = _diffeqdotjl_ode_simulate_inner(
        dynamics, initial_state, timespan, atemp_params=atemp_params, **kwargs
    )

    # TODO support dim != -1
    idx_name = "__time"
    name_to_dim = {k: f.dim - 1 for k, f in get_index_plates().items()}
    name_to_dim[idx_name] = -1

    final_idx = IndexSet(**{idx_name: {len(timespan) - 1}})
    final_state_traj = gather(trajectory, final_idx, name_to_dim=name_to_dim)
    final_state = _squeeze_time_dim(final_state_traj)
    return final_state
