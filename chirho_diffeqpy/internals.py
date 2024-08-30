import functools
import numbers
from functools import singledispatch
from math import prod
from copy import copy
from typing import (
    Callable,
    Dict,
    Hashable,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import juliacall
import numpy as np
import pyro
import torch
from chirho.dynamical.handlers.interruption import StaticEvent, ZeroEvent
from chirho.dynamical.internals._utils import _var_order
from chirho.dynamical.internals.solver import Interruption, simulate_point
from chirho.dynamical.ops import State
from chirho.indexed.ops import IndexSet, gather, get_index_plates, indices_of
from diffeqpy import de
from juliatorch import JuliaFunction
from torch import Tensor as Tnsr

from chirho_diffeqpy.lang_interop import callable_from_julia

from .load_julia_env import load_julia_env

jl = load_julia_env()

jl.seval("using Symbolics")

T = TypeVar("T")
R = Union[numbers.Real, T]
ATempParams = Mapping[str, T]
PureDynamics = Callable[[State[T], ATempParams[T]], State[T]]
PureEventFn = Callable[[R, State[T], ATempParams[T]], R]

_ShapeType = Tuple[int, ...]
_NamedShape = Tuple[str, _ShapeType]
MappingShape = Tuple[_NamedShape, ...]


def get_mapping_shape(mapping: Mapping[str, Tnsr]) -> MappingShape:
    var_order = get_var_order(mapping)
    return tuple((_var, mapping[_var].shape) for _var in var_order)


@functools.singledispatch
def to_numpy(v):
    raise NotImplementedError()


@to_numpy.register
def _(v: Tnsr):
    return v.detach().numpy()


@to_numpy.register
def _(v: dict):
    return {k: to_numpy(v) for k, v in v.items()}


@to_numpy.register
def _(v: tuple):
    return tuple(to_numpy(v) for v in v)


@to_numpy.register
def _(v: np.ndarray):
    return v


def pre_broadcast_initial_state(
    f: Callable, initial_state: State[torch.Tensor], *args, **kwargs
) -> State[torch.Tensor]:
    assert len(initial_state.keys()), "Initial state must have at least one key."

    # Convert to numpy arrays.
    args_np = to_numpy(args)
    kwargs_np = to_numpy(kwargs)

    # Tiled elements of the state might not actually fully prebroadcast in one execution of the dynamics function.
    # E.g. if S affects I affects R, and S is tiled, then R won't be fully pre broadcasted after the first run.
    broadcasted_initial_state = initial_state
    outputs = []  # for check below
    for _ in initial_state.keys():
        broadcasted_initial_state_np = to_numpy(broadcasted_initial_state)
        if "t" not in broadcasted_initial_state_np:
            broadcasted_initial_state_np["t"] = 1.0
        output = f(broadcasted_initial_state_np, *args_np, **kwargs_np)
        outputs.append(output)  # for check below
        broadcasted_initial_state = {
            k: initial_state[k] + torch.zeros(output[k].shape)
            for k, v in initial_state.items()
        }

    # TODO only run this in check_dynamics?
    # <Check Pre-Broadcast>
    broadcasted_initial_state_np = to_numpy(broadcasted_initial_state)
    if "t" not in broadcasted_initial_state_np:
        broadcasted_initial_state_np["t"] = 1.0
    output_w_prebroadcast = f(broadcasted_initial_state_np, *args_np, **kwargs_np)
    # Require that the pre-broadcasted output is identical to all iterations of outputs, e.g. with the original
    #  initial state.
    for output in outputs:
        for k, v in output.items():
            assert np.allclose(
                v, output_w_prebroadcast[k]
            ), "pre_broadcast of initial_state resulted in different output!"
    # Also require that the output_w_prebroadcast has the same shape as the broadcasted_initial_state. This tells us
    #  that we have fully pre-broadcasted the initial state (and didn't need any extra iterations, i.e. we converged).
    check_broadcasted_initial_state = {
        k: initial_state[k] + torch.zeros(output_w_prebroadcast[k].shape)
        for k, v in initial_state.items()
    }
    for (_, v1), (_, v2) in zip(
        broadcasted_initial_state.items(), check_broadcasted_initial_state.items()
    ):
        assert (
            v1.shape == v2.shape
        ), "pre_broadcast of initial_state failed to converge!"
    # </Check Pre-Broadcast>

    return broadcasted_initial_state


def diffeqdotjl_compile_problem(
    dynamics: PureDynamics[np.ndarray],
    initial_state: State[Tnsr],
    start_time: Tnsr,
    end_time: Tnsr,
    atemp_params: ATempParams[Tnsr],
) -> de.ODEProblem:
    # TODO take general compilation kwargs here too.

    require_float64(initial_state)
    require_float64(atemp_params)
    require_float64(dict(start_time=start_time, end_time=end_time))

    # See the note below for why this must be a pure function and cannot use the values in initial_atemp_params
    # directly. callable_from_julia(out_as_first_arg) just specifies that it expects julia to call it with the
    #  preallocated output array as the first argument (as opposed to listing out as a keyword, which is the
    #  default expectation).
    # diffeqpy, when compiling the dynamics function, passes the output array as the first argument.
    @callable_from_julia(out_as_first_arg=True)
    def ode_f(flat_state, flat_inner_atemp_params, t):
        # Unflatten the state u according to the state variables stored in initial dstate.
        state: State = _unflatten_mapping(
            flattened_mapping=flat_state, shaped_mapping_for_reference=initial_state
        )

        # Note that initial_atemp_params will be a dictionary of shaped torch tensors, while flat_atemp_params will be
        # a vector of julia symbolics involved in jit copmilation.
        inner_atemp_params: ATempParams = (
            _unflatten_mapping(
                flattened_mapping=flat_inner_atemp_params,
                shaped_mapping_for_reference=atemp_params,
            )
            if len(flat_inner_atemp_params) > 0
            else dict()
        )

        if "t" in state:
            raise ValueError(
                "Cannot have a state variable named 't' as it is reserved for the time variable."
            )
        state = dict(**state, t=t)

        dstate: State = dynamics(state, inner_atemp_params)

        flat_dstate = _flatten_mapping(dstate)

        return flat_dstate

    # Pre-broadcast the initial state so that we compile the dynamics function with the proper state-space dimensions
    #  (as induced, typically, by the dynamics broadcasting the parameters over an unexpanded initial state).
    initial_state = pre_broadcast_initial_state(dynamics, initial_state, atemp_params)

    # Flatten the initial state and parameters.
    flat_initial_state = _flatten_mapping(initial_state)
    flat_atemp_params = _flatten_mapping(atemp_params)

    # See juliatorch readme to motivate the FullSpecialize syntax.
    prob = de.seval("ODEProblem{true, SciMLBase.FullSpecialize}")(
        ode_f,
        # Note that we are only interested in compilation here, not differentiation! These values will strictly be
        #  used for their types and shapes so that the ODEProblem knows what to compile for.
        flat_initial_state.detach().numpy(),
        torch.cat((start_time[None], end_time[None])).detach().numpy(),
        # HACK sgfeh2 stuff breaks with a len zero array, and passing nothing or None incurs the same problem.
        (
            flat_atemp_params.detach().numpy()
            if len(flat_atemp_params)
            else np.ones((1,), dtype=np.float64)
        ),
    )

    fast_prob = de.jit(prob)

    return fast_prob


# TODO g179du91 move to internal ops as other backends might also use this?
# See use in handlers/solver.DiffEqDotJL._pyro__lazily_compile_problem
@pyro.poutine.runtime.effectful(type="_lazily_compile_problem")
def _lazily_compile_problem(
    dynamics: PureDynamics[np.ndarray],
    initial_state: State[Tnsr],
    start_time,
    end_time,
    atemp_params,
) -> de.ODEProblem:
    raise NotImplementedError()


# TODO g179du91
@pyro.poutine.runtime.effectful(type="_lazily_compile_event_fn_callback")
def _lazily_compile_event_fn_callback(
    interruption: Interruption,
    initial_state: State[Tnsr],
    atemp_params: ATempParams[Tnsr],
) -> de.VectorContinuousCallback:
    raise NotImplementedError()


def get_var_order(mapping: Mapping[str, Union[Tnsr, np.ndarray]]) -> Tuple[str, ...]:
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
def _flatten_mapping(
    mapping,  #: Mapping[str, Union[Tnsr, np.ndarray]]  # TODO do17bdy1t
):  # -> Union[Tnsr, np.ndarray]:  # TODO do17bdy1t
    if len(mapping) == 0:
        # TODO do17bdy1t address type specificity
        return torch.tensor([], dtype=torch.float64)
    var_order = get_var_order(mapping)
    return flat_cat(*[mapping[v] for v in var_order])


# TODO ofwr1948 replace with generic pytree flatten/unflatten?
def _unflatten_mapping(
    flattened_mapping: Tnsr,
    shaped_mapping_for_reference: Mapping[str, Union[Tnsr, juliacall.VectorValue]],
    traj_length: Optional[int] = None,
) -> Mapping[str, Union[Tnsr, np.ndarray]]:
    var_order = get_var_order(shaped_mapping_for_reference)
    mapping_to_return = dict()
    for v in var_order:
        shaped = shaped_mapping_for_reference[v]
        shape = shaped.shape
        if traj_length is not None:
            # If this is a trajectory of states, the dimension following the original shape's state will be time,
            #  and because we know the rest of the shape we can auto-detect its size with -1.
            shape += (traj_length,)

        sv = flattened_mapping[: prod(shape)].reshape(shape)

        mapping_to_return[v] = sv

        # Slice so that only the remaining elements are left.
        flattened_mapping = flattened_mapping[prod(shape) :]

    return mapping_to_return


# TODO ofwr1948 replace with generic pytree traversal.
def require_float64(mapping: Mapping[str, Tnsr]):
    # Forward diff through diffeqpy currently requires float64. # TODO do17bdy1t update when this is fixed.
    for k, v in mapping.items():
        if v.dtype is not torch.float64:
            raise ValueError(
                f"State or parameter variable {k} has dtype {v.dtype}, but must be float64."
            )


def _diffeqdotjl_ode_simulate_inner(
    dynamics: PureDynamics[np.ndarray],
    initial_state: State[Tnsr],
    timespan: Tnsr,
    atemp_params: ATempParams[Tnsr],
    _diffeqdotjl_callback: Optional[de.CallbackSet] = None,
    **kwargs,
) -> Tuple[State[torch.Tensor], State[torch.Tensor], torch.Tensor]:
    # if not torch.all(timespan[:-1] < timespan[1:]):
    #     raise ValueError("The requested times must be sorted and strictly increasing.")

    if not isinstance(atemp_params, dict):
        raise ValueError(
            f"atemp_params must be a dictionary, "
            f"but got type {type(atemp_params)} and value {atemp_params}"
        )

    require_float64(initial_state)
    require_float64(atemp_params)
    require_float64(dict(timespan=timespan))

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
    )

    initial_state = pre_broadcast_initial_state(dynamics, initial_state, atemp_params)

    # Flatten the initial state, timespan, and parameters into a single vector. This is required because
    #  juliatorch currently requires a single matrix or vector as input.
    flat_initial_state = _flatten_mapping(initial_state)
    flat_atemp_params = _flatten_mapping(atemp_params)
    # HACK sgfeh2 stuff breaks with a len zero array, and passing nothing or None incurs the same problem.
    flat_atemp_params = (
        torch.ones((1,), dtype=torch.float64)
        if len(flat_atemp_params) == 0
        else flat_atemp_params
    )
    outer_u0_t_p = torch.cat([flat_initial_state, timespan, flat_atemp_params])

    # This has to take a concatenated vector because of juliacall's one input/one output limitations.
    def inner_solve(u0_t_p):

        # Unpack the concatenated initial state, timespan, and parameters.
        u0 = u0_t_p[: flat_initial_state.numel()]
        tspan = u0_t_p[flat_initial_state.numel() : -flat_atemp_params.numel()]
        p = u0_t_p[-flat_atemp_params.numel() :]

        # Remake the otherwise-immutable problem to use the new parameters.
        remade_compiled_prob = de.remake(
            compiled_prob, u0=u0, p=p, tspan=(tspan[0], tspan[-1])
        )

        kwargs_copy = copy(kwargs)
        alg = kwargs_copy.pop("alg", None)

        # Passing saveat means we won't save dense higher order interpolants. This saves on memory, and still lets us
        #  evaluate at the requested times later, as if interpolating.
        sol = de.solve(
            remade_compiled_prob,
            alg,
            callback=_diffeqdotjl_callback,
            saveat=tspan,
            **kwargs_copy
        )

        # Get the end time of the trajectory. Note that this may precede elements in the tspan if the solver
        #  was terminated by a dynamic event.
        inner_end_t = sol.t[-1]

        # Interpolate at the requested times, including at the final state. So, add the terminal solve time to the
        #  timespan.
        jl.tspan = tspan
        jl.inner_end_t = inner_end_t
        jl.seval("augmented_tspan = vcat(tspan, inner_end_t)")
        augmented_tspan = jl.augmented_tspan

        inner_flat_traj: juliacall.ArrayValue = sol(augmented_tspan)

        # We now need to
        # 1) fully flatten the traj
        # 2) concatenate the end_t to the end of the traj
        # 3) return the concatenated vector
        # This is required because JuliaFunction.apply only supports functions with single tensor outputs, and
        #  we need the end time returned here to a) differentiate wrt it and b) know what time vals we should be
        #  returning state for (e.g. not the ones that happen after termination)
        jl.inner_flat_traj = inner_flat_traj
        jl.inner_end_t = inner_end_t

        # inner_flat_flat_traj will be a vector of doubles or forward diff duals.
        # inner_end_t will be a double or forward diff dual.
        # cat them using native julia.
        # julia uses column major ordering, but we'll be unflattening assuming row major, so invert the axes before
        #  flattening.
        jl.seval("row_major_perm = reverse(1:ndims(inner_flat_traj))")
        jl.seval(
            "inner_flat_flat_traj = vec(permutedims(inner_flat_traj, row_major_perm))"
        )
        jl.seval("out_jl = vcat(inner_flat_flat_traj, inner_end_t)")

        return jl.out_jl

    # Finally, execute the juliacall function
    solve_result = JuliaFunction.apply(inner_solve, outer_u0_t_p)
    # Extract out the concatenated return values.
    flat_traj = solve_result[..., :-1]
    end_t = solve_result[-1]

    # Unflatten the trajectory.
    # return _unflatten_state(flat_traj, initial_state, to_traj=True)
    traj = _unflatten_mapping(
        flattened_mapping=flat_traj,
        shaped_mapping_for_reference=initial_state,
        traj_length=len(timespan)
        + 1,  # because augmented the timespan with the end state.
    )

    requested_traj = {k: v[..., :-1] for k, v in traj.items()}
    final_state = {k: v[..., -1] for k, v in traj.items()}

    # If no dynamic interventions could have fired, just return the traj.
    if _diffeqdotjl_callback is None:
        return requested_traj, final_state, end_t

    # If a dynamic event was fired, then end_t will precede logging elements in the timespan, so sol(tspan) will just
    # return the end state for any interpolation times that preceeded the end time. We don't want to return anything
    # for those times, so we need to
    #  1) find the timespan elements that are less than or equal to the interruption time (end_t)
    #  2) return the requested_traj after masking out those states logged after end_t.
    pre_event_mask = timespan < end_t
    pre_event_requested_traj = {
        k: v[..., pre_event_mask] for k, v in requested_traj.items()
    }

    return pre_event_requested_traj, final_state, end_t


def diffeqdotjl_simulate_trajectory(
    dynamics: PureDynamics[np.ndarray],
    initial_state: State[Tnsr],
    timespan: Tnsr,
    **kwargs,
) -> State[Tnsr]:
    requested_traj, final_State, end_t = _diffeqdotjl_ode_simulate_inner(
        dynamics, initial_state, timespan, **kwargs
    )
    return requested_traj


def diffeqdotjl_simulate_to_interruption(
    interruptions: List[Interruption],
    dynamics: PureDynamics[np.ndarray],
    initial_state: State[Tnsr],
    start_time: Tnsr,
    end_time: Tnsr,
    atemp_params: ATempParams[Tnsr],
    **kwargs,
) -> Tuple[State[Tnsr], Tnsr, Optional[Interruption]]:
    # Static interruptions can be handled statically, so sort out dynamics from statics.
    dynamic_interruptions = [
        interruption
        for interruption in interruptions
        if not isinstance(interruption.predicate, StaticEvent)
    ]
    static_times = torch.stack(
        [
            (
                interruption.predicate.time
                if isinstance(interruption.predicate, StaticEvent)
                else torch.tensor(torch.inf)
            )
            for interruption in interruptions
        ]
    )

    static_time_min_idx = torch.argmin(static_times)
    static_end_time = static_times[static_time_min_idx]
    assert torch.isfinite(
        static_end_time
    ), "Internal error: static_end_time should be finite. Is end_time non-finite?"
    static_end_interruption = interruptions[static_time_min_idx]

    if len(dynamic_interruptions) == 0:
        final_state = simulate_point(
            dynamics,
            initial_state,
            start_time,
            static_end_time,
            atemp_params=atemp_params,
            **kwargs,
        )
        return final_state, static_end_time, static_end_interruption

    initial_state = pre_broadcast_initial_state(dynamics, initial_state, atemp_params)
    cb = _diffeqdotjl_build_combined_event_f_callback(
        dynamic_interruptions, initial_state=initial_state, atemp_params=atemp_params
    )

    # FIXME iwgar0kgs72 So the original backend implementation for TorchDiffEq assumes that neither simulate_point
    #  nor (implicitly b/c of how it's used in LogTrajectory) simulate_trajectory are interruptible. The has the
    #  unfortunate side effect of requiring that dynamic intervention times be found before the trajectory is logged.
    #  This design was landed on b/c torchdiffeq cannot solve for intermediate time points while also searching for
    #  an interruption. diffeqpy, however, can. We should refactor "upstream" abstractions in favor of this latter
    #  functioning, rather than the former induced by weird torchdiffeq limitations.
    #  For now though, we'll proceed as we do with torchdiffeq and find the interruption time first, and then resolve
    #  with simulate_point in order to log the trajectory.

    # TODO HACK 18wfghjfs541 Set global interruption callback pointer to None.
    #  This will be set to the triggering interruption if one is found.
    #  Maybe make this local in _diffeqdotjl_ode_simulate_inner somehow? This is hard because the event_fns
    #  themselves get lazily compiled wrt this pointer.
    _last_triggered_interruption_ptr[0] = None

    requested_trajectory, final_state, interruption_time = (
        _diffeqdotjl_ode_simulate_inner(
            dynamics,
            initial_state,
            torch.stack((start_time, static_end_time)),
            atemp_params=atemp_params,
            _diffeqdotjl_callback=cb,
            **kwargs,
        )
    )

    # If no dynamic intervention's affect! function fired, then the static interruption is responsible for
    #  termination.
    # TODO HACK 18wfghjfs541
    if _last_triggered_interruption_ptr[0] is None:
        triggering_interruption = static_end_interruption
        interruption_time = static_end_time
    else:
        triggering_interruption = _last_triggered_interruption_ptr[0]

    # FIXME iwgar0kgs72 simulate a second time with the effectful op to log the trajectory.
    #  See above for why this is silly. Note that we aren't using anything returned here...
    simulate_point(
        dynamics,
        initial_state,
        start_time,
        interruption_time,
        atemp_params=atemp_params,
        # _diffeqdotjl_callback=cb,  # FIXME iwgar0kgs72 ignoring dynamic interruption callbacks here.
        **kwargs,
    )

    return final_state, interruption_time, triggering_interruption


def diffeqdotjl_simulate_point(
    dynamics: PureDynamics[np.ndarray],
    initial_state: State[torch.Tensor],
    start_time: torch.Tensor,
    end_time: torch.Tensor,
    atemp_params: ATempParams[torch.Tensor],
    **kwargs,
) -> State[torch.Tensor]:
    timespan = torch.stack((start_time, end_time))
    requested_trajectory, final_state, end_t = _diffeqdotjl_ode_simulate_inner(
        dynamics, initial_state, timespan, atemp_params=atemp_params, **kwargs
    )

    return final_state


@functools.singledispatch
def numel(x):
    raise NotImplementedError(f"numel not implemented for type {type(x)}.")


@numel.register
def _(x: np.ndarray):
    return x.size


@numel.register
def _(x: Tnsr):
    return x.numel()


@numel.register
def _(x: juliacall.ArrayValue):
    return prod(x.shape)


@numel.register
def _(x: float):
    return 1


def diffeqdotjl_compile_event_fn_callback(
    interruption: Interruption,
    initial_state: State[Tnsr],
    atemp_params: ATempParams[Tnsr],
) -> de.VectorContinuousCallback:
    if not isinstance(interruption.predicate, ZeroEvent):
        raise ValueError(
            "event_fn compilation received interruption with unexpected predicate (not a ZeroEvent)."
            " Only ZeroEvents can currently be compiled."
        )

    if isinstance(interruption.predicate, StaticEvent):
        # The torchdiffeq backend only supports dynamic xor static termination, but diffeqpy supports dynamic
        #  termination with a static end time specified, so we don't need to treat static events dynamically.
        raise ValueError(
            "event_fn compilation for the DiffEqPy backend should not have to be applied to StaticEvents,"
            " as those events can be handled statically by specifying an end_time, alongside any"
            " relevant dynamic events passed in a callback set."
        )

    event_fn: PureEventFn = interruption.predicate.event_fn

    # Execute the event_fn once to get the shape of the output.
    try:
        ret1 = event_fn(0.0, to_numpy(initial_state), to_numpy(atemp_params))
    except TypeError as e:
        if "takes 2 positional arguments but 3 were given" in str(e):
            raise ValueError(
                "event_fn for use with the DiffEqPy backend must take both state and parameters as"
                " arguments."
            )
    numel_out = numel(ret1)

    # Define the inner bit of the condition function that we're going to compile.
    @callable_from_julia(out_as_first_arg=True)
    def inner_condition_(u, t, p):

        state = _unflatten_mapping(
            flattened_mapping=u, shaped_mapping_for_reference=initial_state
        )

        params = _unflatten_mapping(
            flattened_mapping=p, shaped_mapping_for_reference=atemp_params
        )

        ret = event_fn(t, state, params)

        # ret could be a scalar, so put it in an array, and then ravel to support unwrapping to the output vector.
        return np.array(ret).ravel()

    flat_initial_state = _flatten_mapping(initial_state)
    flat_atemp_params = _flatten_mapping(atemp_params)

    # Define symbolic inputs to inner_condition_. The JuliaThingWrapper machinery doesn't support
    #  symbolic arrays though because indexing operations are difficult to forward. So these need to be non-symbolic
    #  vectors of symbols. This is achieved via "scalarize".
    jl.seval(
        f"@variables uvec[1:{len(flat_initial_state)}], t, pvec[1:{len(flat_atemp_params)}]"
    )
    jl.seval("u = Symbolics.scalarize(uvec)")
    jl.seval("p = Symbolics.scalarize(pvec)")
    # Make just a generic empty output vector of type Num with length numel_out.
    jl.seval(f"out = [Num(0) for _ in 1:{numel_out}]")

    # Symbolically evaluate the inner_condition_ for the resultant expression.
    inner_condition_(jl.out, jl.u, jl.t, jl.p)

    # Build the inner_condition_ function.
    built_expr = jl.seval("build_function(out, u, t, p)")

    # TODO HACK figure out a less janky namespace solution. maybe just write this compilation in julia and call
    #  it here.
    import uuid

    affect_uuid = uuid.uuid4().hex

    # Evaluate it to turn it into a julia function.
    # This builds both an in place and regular function, but we only need the in place one.
    assert len(built_expr) == 2
    # jl.inner_condition_ = jl.eval(built_expr[-1])
    setattr(jl, f"inner_condition_{affect_uuid}", jl.eval(built_expr[-1]))

    # inner_condition can now be called from a condition function with signature
    #  expected by the callbacks.
    jl.seval(
        f"""
    function condition_{affect_uuid}(out, u, t, integrator)
        inner_condition_{affect_uuid}(out, u, t, integrator.p)
    end
    """
    )

    # The "affect" function is only called a single time, so we can just use python. This
    #  function also tracks which interruption triggered the termination.
    def affect_b(integrator, *_):
        # TODO HACK 18wfghjfs541 using a global "last interruption" is meh, but using the affect function
        #  to directly track which interruption was responsible for termination is a lot cleaner than running
        #  the event_fns after the fact to figure out which one was responsible.
        _last_triggered_interruption_ptr[0] = interruption
        de.terminate_b(integrator)

    # Return the callback involving only juila functions.
    return de.VectorContinuousCallback(
        # jl.condition_,
        getattr(jl, f"condition_{affect_uuid}"),
        affect_b,
        numel_out,
    )


# TODO HACK 18wfghjfs541
# Can this go in the solver somehow so we don't have a global?
_last_triggered_interruption_ptr = [None]  # type: List[Optional[Interruption]]


def _diffeqdotjl_build_combined_event_f_callback(
    interruptions: List[Interruption],
    initial_state: State[Tnsr],
    atemp_params: ATempParams[Tnsr],
) -> de.CallbackSet:
    cbs = []

    for i, interruption in enumerate(interruptions):
        vc_cb = _lazily_compile_event_fn_callback(
            interruption, initial_state, atemp_params
        )  # type: de.VectorContinuousCallback

        cbs.append(vc_cb)

    return de.CallbackSet(*cbs)


@indices_of.register
def _indices_of_ndarray(value: np.ndarray, **kwargs) -> IndexSet:
    return indices_of(value.shape, **kwargs)


def _index_select_from_ndarray(
    arr: np.ndarray, dim: int, indices: List[int]
) -> np.ndarray:
    return np.take(arr, indices=np.array(indices), axis=dim)


# Note dg1ofo9: This implementation is identical to the torch tensor implementation, except that it uses
#  _index_select_from_ndarray above.
@gather.register
def _gather_ndarray(
    value: np.ndarray,
    indexset: IndexSet,
    *,
    event_dim: Optional[int] = None,
    name_to_dim: Optional[Dict[Hashable, int]] = None,
    **kwargs,
) -> np.ndarray:
    if event_dim is None:
        event_dim = 0
    if name_to_dim is None:
        name_to_dim = {name: f.dim for name, f in get_index_plates().items()}
    result = value
    for name, indices in indexset.items():
        if name not in name_to_dim:
            continue
        dim = name_to_dim[name] - event_dim
        if len(result.shape) < -dim or result.shape[dim] == 1:
            continue
        result = _index_select_from_ndarray(
            result, name_to_dim[name] - event_dim, list(sorted(indices))
        )
        # See above for note dg1ofo9
        # result = result.index_select(
        #     name_to_dim[name] - event_dim,
        #     torch.tensor(list(sorted(indices)), device=value.device, dtype=torch.long),
        # )
    return result


class DiffEqPyRuntimeCheck(pyro.poutine.messenger.Messenger):
    def _pyro_sample(self, msg):
        raise ValueError(
            "The chirho DiffEqPy backend currently only supports ODE models and not SDE models,"
            " and thus does not allow `pyro.sample` calls within the dynamics."
        )


def diffeqpy_check_dynamics(
    dynamics: PureDynamics[np.ndarray],
    initial_state: State[torch.Tensor],
    start_time: torch.Tensor,
    end_time: torch.Tensor,
    atemp_params: ATempParams[torch.Tensor],
) -> None:
    # Pre-broadcast (this also checks that the prebroadcast reaches a shape fixed point).
    expanded_initial_state = pre_broadcast_initial_state(
        dynamics, initial_state, atemp_params
    )

    expanded_initial_state_np = to_numpy(expanded_initial_state)
    atemp_params_np = to_numpy(atemp_params)

    with DiffEqPyRuntimeCheck():
        dt_np = dynamics(expanded_initial_state_np, atemp_params_np)

    if not isinstance(dt_np, dict):
        raise ValueError("Dynamics function must return a dictionary of dstates.")

    if dt_np.keys() != expanded_initial_state_np.keys():
        raise ValueError(
            f"Time derivative for state variables {dt_np.keys()} does not match the state variables "
            f"{expanded_initial_state_np.keys()}."
        )

    for k, v in dt_np.items():
        if v.shape != expanded_initial_state_np[k].shape:
            raise ValueError(
                f"Time derivative for state variable {k} has shape {v.shape}, but expected it be the same "
                f" shape as the state {expanded_initial_state_np[k].shape}."
            )

        if isinstance(v, torch.Tensor):
            raise NotImplementedError(
                f"Time derivative for state variable {k} is a torch tensor, but currently dynamics"
                " must be defined to operate on and return numpy arrays."
            )
