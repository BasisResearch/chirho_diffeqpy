from typing import Callable, List, Optional, Tuple

import pyro
import torch
from diffeqpy import de

from chirho.dynamical.internals._utils import _squeeze_time_dim, _var_order
from chirho.dynamical.internals.solver import Interruption, simulate_point
from chirho.dynamical.ops import Dynamics, State as StateAndOrParams
from chirho.indexed.ops import IndexSet, gather, get_index_plates
from torch import Tensor as Tnsr
from juliatorch import JuliaFunction
import juliacall
import numpy as np
from typing import Union
from functools import singledispatch
from copy import copy
from juliacall import Main as jl


def diffeqdotjl_compile_problem(
    dynamics: Dynamics[np.ndarray],
    initial_state_and_params: StateAndOrParams[Tnsr],
    start_time: Tnsr,
    end_time: Tnsr,
    **kwargs
) -> de.ODEProblem:

    require_float64(initial_state_and_params)

    initial_state, torch_params = separate_state_and_params(dynamics, initial_state_and_params, start_time)

    # See the note below for why this must be a pure function and cannot use the values in torch_params directly.
    def ode_f(flat_dstate_out, flat_state, flat_params, t):
        # Unflatten the state u according to the state variables stored in initial dstate.
        state = _unflatten_state(
            # Wrap julia symbolics (that will be passed through this during jit) so that numpy doesn't introspect them
            #  as sequences with a large number of dimensions.
            JuliaThingWrapper.wrap_array(flat_state),
            initial_state
        )

        # Note that initial_params will be a dictionary of shaped torch tensors, while flat_params will be a vector
        # of julia symbolics involved in jit copmilation. I.e. while initial_params has the same real values as
        # flat_params, they do not carry gradient information that can be propagated through the julia solver.
        params = _unflatten_state(
            JuliaThingWrapper.wrap_array(flat_params),
            torch_params
        ) if len(flat_params) > 0 else StateAndOrParams()

        state_ao_params = StateAndOrParams(**state, **params, t=JuliaThingWrapper(t))

        dstate = dynamics(state_ao_params)

        flat_dstate = _flatten_state_ao_params(dstate)

        try:
            # Unwrap the array of JuliaThingWrappers back into a numpy array of julia symbolics.
            JuliaThingWrapper.unwrap_array(flat_dstate, out=flat_dstate_out)
        except IndexError as e:
            # TODO this could be made more informative by pinpointing which particular dstate is the wrong shape.
            raise IndexError(f"Number of elements in dstate ({len(flat_dstate)}) does not match the number of"
                             f" elements defined in the initial state ({len(flat_dstate_out)}). "
                             f"\nOriginal error: {e}")

    # Flatten the initial state and parameters.
    flat_initial_state = _flatten_state_ao_params(initial_state)
    flat_torch_params = _flatten_state_ao_params(torch_params)

    # See juliatorch readme to motivate the FullSpecialize syntax.
    prob = de.seval("ODEProblem{true, SciMLBase.FullSpecialize}")(
        ode_f,
        flat_initial_state.detach().numpy(),
        np.array([start_time, end_time], dtype=np.float64),
        flat_torch_params.detach().numpy())

    fast_prob = de.jit(prob)

    return fast_prob


# TODO g179du91 move to internal ops as other backends might also use this?
# See use in handlers/solver.DiffEqDotJL._pyro__lazily_compile_problem
@pyro.poutine.runtime.effectful(type="_lazily_compile_problem")
def _lazily_compile_problem(*args, **kwargs) -> de.ODEProblem:
    raise NotImplementedError()


def get_var_order(state_ao_params: StateAndOrParams[Tnsr]) -> Tuple[str, ...]:
    return _var_order(frozenset(state_ao_params.keys()))


# Single dispatch cat thing that handles both tensors and numpy arrays.
@singledispatch
def flat_cat(*vs):
    # Default implementation assumes we're dealing some underying julia thing that needs to be put back into an array.
    # This will re-route to the numpy implementation.

    # If atleast_1d receives a single argument, it will return a single array, rather than a tuple of arrays.
    vs = np.atleast_1d(*vs)
    return flat_cat_numpy(*(vs if isinstance(vs, list) else (vs,)))


@flat_cat.register
def flat_cat_torch(*vs: Tnsr):
    return torch.cat([v.ravel() for v in vs])


@flat_cat.register
def flat_cat_numpy(*vs: np.ndarray):
    return np.concatenate([v.ravel() for v in vs])


def _flatten_state_ao_params(state_ao_params: StateAndOrParams[Union[Tnsr, np.ndarray]]) -> Union[Tnsr, np.ndarray]:
    if len(state_ao_params) == 0:
        # TODO do17bdy1t address type specificity
        return torch.tensor([], dtype=torch.float64)
    var_order = get_var_order(state_ao_params)
    return flat_cat(*[state_ao_params[v] for v in var_order])


def _unflatten_state(
        flat_state_ao_params: Tnsr,
        shaped_state_ao_params: StateAndOrParams[Union[Tnsr, juliacall.VectorValue]],
        to_traj: bool = False
) -> StateAndOrParams[Union[Tnsr, np.ndarray]]:

    var_order = get_var_order(shaped_state_ao_params)
    state_ao_params = StateAndOrParams()
    for v in var_order:
        shaped = shaped_state_ao_params[v]
        shape = shaped.shape
        if to_traj:
            # If this is a trajectory of states, the dimension following the original shape's state will be time,
            #  and because we know the rest of the shape we can auto-detect its size with -1.
            shape += (-1,)

        sv = flat_state_ao_params[:shaped.numel()].reshape(shape)

        state_ao_params[v] = sv

        # Slice so that only the remaining elements are left.
        flat_state_ao_params = flat_state_ao_params[shaped.numel():]

    return state_ao_params


def require_float64(state_ao_params: StateAndOrParams[Tnsr]):
    # Forward diff through diffeqpy currently requires float64. # TODO do17bdy1t update when this is fixed.
    for k, v in state_ao_params.items():
        if v.dtype is not torch.float64:
            raise ValueError(f"State variable {k} has dtype {v.dtype}, but must be float64.")


def separate_state_and_params(dynamics: Dynamics[np.ndarray], initial_state_ao_params: StateAndOrParams[Tnsr], t0: Tnsr):
    """
    Non-explicitly (bad?), the initial_state must include parameters that inform dynamics. This is required
     for this backend because the solve function passed to Julia must be a pure wrt to parameters that
     one wants to differentiate with respect to.
    """

    # Copy so we can add time in without modifying the original, also convert elements to numpy arrays so that the
    #  user's dynamics only have to handle numpy arrays, and not also torch tensors. This is fine, as the only way
    #  we use the initial_dstate below is for its keys, which comprise the state variables.
    initial_state_ao_params_np = {k: copy(v.detach().numpy()) for k, v in initial_state_ao_params.items()}
    # TODO unify this time business with how torchdiffeq is doing it?
    if 't' in initial_state_ao_params_np:
        raise ValueError("Initial state cannot contain a time variable 't'. This is added on the backend.")
    initial_state_ao_params_np['t'] = t0.detach().numpy()

    # Run the dynamics on the converted initial state.
    initial_dstate_np = dynamics(initial_state_ao_params_np)

    # Keys that don't appear in the returned dstate are parameters.
    param_keys = [k for k in initial_state_ao_params.keys() if k not in initial_dstate_np.keys()]
    # Keys that do appear in the dynamics are state variables.
    state_keys = [k for k in initial_state_ao_params.keys() if k in initial_dstate_np.keys()]

    torch_params = StateAndOrParams(**{k: initial_state_ao_params[k] for k in param_keys})
    initial_state = StateAndOrParams(**{k: initial_state_ao_params[k] for k in state_keys})

    return initial_state, torch_params


def _diffeqdotjl_ode_simulate_inner(
    dynamics: Dynamics[np.ndarray],
    initial_state_and_params: StateAndOrParams[Tnsr],
    timespan: Tnsr,
    **kwargs
) -> StateAndOrParams[torch.tensor]:

    require_float64(initial_state_and_params)

    # The backend solver requires that the dynamics are a pure function, meaning the parameters must be passed
    #  in as arguments. Thus, we simply require that the params are passed along in the initial state, and assume
    #  that anything not returned by the dynamics are parameters, and not state.
    initial_state, torch_params = separate_state_and_params(dynamics, initial_state_and_params, timespan[0])

    # Flatten the initial state, timespan, and parameters into a single vector. This is required because
    #  juliatorch currently requires a single matrix or vector as input.
    flat_initial_state = _flatten_state_ao_params(initial_state)
    flat_torch_params = _flatten_state_ao_params(torch_params)
    outer_u0_t_p = torch.cat([flat_initial_state, timespan, flat_torch_params])

    compiled_prob = _lazily_compile_problem(
        dynamics,
        # Note: these inputs are only used on the first compilation so that the types, shapes etc. get compiled along
        # with the problem. Subsequently (in the inner_solve), these are ignored (even though they have the same as
        # exact values as the args passed into `remake` below). The outer_u0_t_p has to be passed into the
        # JuliaFunction.apply so that those values can be put into Dual numbers by juliatorch.
        initial_state_and_params,
        timespan[0],
        timespan[-1],
        **kwargs,
    )

    def inner_solve(u0_t_p):

        # Unpack the concatenated initial state, timespan, and parameters.
        u0 = u0_t_p[:flat_initial_state.numel()]
        tspan = u0_t_p[flat_initial_state.numel():-flat_torch_params.numel()]
        p = u0_t_p[-flat_torch_params.numel():]

        # Remake the otherwise-immutable problem to use the new parameters.
        remade_compiled_prob = de.remake(compiled_prob, u0=u0, p=p, tspan=(tspan[0], tspan[-1]))

        sol = de.solve(remade_compiled_prob)

        # Interpolate the solution at the requested times.
        return sol(tspan)

    # Finally, execute the juliacall function
    flat_traj = JuliaFunction.apply(inner_solve, outer_u0_t_p)

    # Unflatten the trajectory.
    return _unflatten_state(flat_traj, initial_state, to_traj=True)


def diffeqdotjl_simulate_trajectory(
    dynamics: Dynamics[np.ndarray],
    initial_state_and_params: StateAndOrParams[Tnsr],
    timespan: Tnsr,
    **kwargs,
) -> StateAndOrParams[Tnsr]:
    return _diffeqdotjl_ode_simulate_inner(dynamics, initial_state_and_params, timespan)


def diffeqdotjl_simulate_to_interruption(
    interruptions: List[Interruption],
    dynamics: Dynamics[np.ndarray],
    initial_state_and_params: StateAndOrParams[Tnsr],
    start_time: Tnsr,
    end_time: Tnsr,
    **kwargs,
) -> Tuple[StateAndOrParams[Tnsr], Tnsr, Optional[Interruption]]:

    # TODO TODO implement the actual retrieval of the next interruption (see torchdiffeq_simulate_to_interruption)

    from chirho.dynamical.handlers.interruption import StaticInterruption
    next_interruption = StaticInterruption(end_time)

    value = simulate_point(
        dynamics, initial_state_and_params, start_time, end_time, **kwargs
    )

    return value, end_time, next_interruption


def diffeqdotjl_simulate_point(
    dynamics: Dynamics[np.ndarray],
    initial_state_and_params: StateAndOrParams[torch.Tensor],
    start_time: torch.Tensor,
    end_time: torch.Tensor,
    **kwargs,
) -> StateAndOrParams[torch.Tensor]:
    # TODO this is exactly the same as torchdiffeq, so factor out to utils or something.

    timespan = torch.stack((start_time, end_time))
    trajectory = _diffeqdotjl_ode_simulate_inner(
        dynamics, initial_state_and_params, timespan, **kwargs
    )

    # TODO support dim != -1
    idx_name = "__time"
    name_to_dim = {k: f.dim - 1 for k, f in get_index_plates().items()}
    name_to_dim[idx_name] = -1

    final_idx = IndexSet(**{idx_name: {len(timespan) - 1}})
    final_state_traj = gather(trajectory, final_idx, name_to_dim=name_to_dim)
    final_state = _squeeze_time_dim(final_state_traj)
    return final_state