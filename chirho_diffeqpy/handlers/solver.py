from __future__ import annotations

from copy import copy
from typing import Dict, Tuple, TypeVar, Callable

import numpy as np
from chirho.dynamical.internals.solver import Interruption, Solver, State  # noqa: F401
from diffeqpy import de  # noqa: F401
from torch import Tensor as Tnsr

from ..internals import (
    ATempParams,
    MappingShape,
    PureDynamics,
    PureEventFn,
    get_mapping_shape,
    pre_broadcast_initial_state,
)

S = TypeVar("S")
T = TypeVar("T")

ATEMPPARAMS_KEY = "atemp_params"

MappingShapePair = Tuple[MappingShape, MappingShape]


class DiffEqPy(Solver[Tnsr]):

    # These match the torchdiffeq defaults in chirho 2.0
    DEFAULT_KWARGS = dict(
        reltol=1e-7,
        abstol=1e-9,
    )

    def __init__(self, **kwargs):
        super().__init__()

        # Add solver arguments to init sig and put them here as needed. They will be passed along below.
        # Note that the passed kwargs will take preference under this ordering.
        self.solve_kwargs = {**self.DEFAULT_KWARGS, **kwargs}

        # Lazily compile solvers used in this context. Compilations are organized first by the dynamics function
        #  object itself, and then by the shapes of the (prebroadcasted) state and atemp_params that have been
        #  compiled for.
        self._lazily_compiled_solvers: Dict[
            PureDynamics[np.ndarray], Dict[MappingShapePair, de.ODEProblem]
        ] = dict()

        # Lazily compile event fns used in this context. Compilations are organized first by the interruption object
        #  and then, as above, by the shapes of the (prebroadcasted) state and atemp_params.
        self._lazily_compiled_event_fns: Dict[
            PureEventFn[np.ndarray], Dict[MappingShapePair, Tuple[Callable, int]]
        ] = dict()

    def _get_or_create_compilations_for_dynamics(
        self, dynamics: PureDynamics[np.ndarray]
    ):
        if dynamics not in self._lazily_compiled_solvers:
            self._lazily_compiled_solvers[dynamics] = dict()
        return self._lazily_compiled_solvers[dynamics]

    def _get_or_create_compilations_for_event_fns(
        self, event_fn: PureEventFn[np.ndarray]
    ):
        if event_fn not in self._lazily_compiled_event_fns:
            self._lazily_compiled_event_fns[event_fn] = dict()
        return self._lazily_compiled_event_fns[event_fn]

    @staticmethod
    def _get_atemp_params_from_msg(msg) -> Tuple[ATempParams, Dict]:
        # We don't modify the msg["kwargs"] in case some outer process is expecting it to remain constant.
        kwargs = copy(msg["kwargs"])
        # But the kwargs that we want to pass on to our stuff will need to have the params extracted,
        #  primarily for clarity.
        atemp_params = kwargs.pop(ATEMPPARAMS_KEY, None)
        if atemp_params is None:
            raise ValueError(
                "DiffEqPy requires that dynamics are implemented as pure functions, which require"
                " that parameters are passed into the simulate call alongside the dynamics function."
                f" Add a '{ATEMPPARAMS_KEY}' key to the kwargs of the simulate call, and ensure your "
                f" dynamics function accepts a '{ATEMPPARAMS_KEY}' argument after the state."
            )
        return atemp_params, kwargs

    def _process_simulate_args_kwargs(self, msg):
        atemp_params, kwargs = self._get_atemp_params_from_msg(msg)
        kwargs.update(self.solve_kwargs)
        return msg["args"], atemp_params, kwargs

    def _pyro_simulate_point(self, msg) -> None:
        from chirho_diffeqpy.internals import diffeqdotjl_simulate_point

        (dynamics, initial_state, start_time, end_time), atemp_params, kwargs = (
            self._process_simulate_args_kwargs(msg)
        )

        msg["value"] = diffeqdotjl_simulate_point(
            dynamics,
            initial_state,
            start_time,
            end_time,
            atemp_params=atemp_params,
            **kwargs,
        )
        msg["done"] = True

    def _pyro_simulate_trajectory(self, msg) -> None:
        from chirho_diffeqpy.internals import diffeqdotjl_simulate_trajectory

        (dynamics, initial_state, timespan), atemp_params, kwargs = (
            self._process_simulate_args_kwargs(msg)
        )

        msg["value"] = diffeqdotjl_simulate_trajectory(
            dynamics, initial_state, timespan, atemp_params=atemp_params, **kwargs
        )
        msg["done"] = True

    def _pyro_simulate_to_interruption(self, msg) -> None:
        from chirho_diffeqpy.internals import diffeqdotjl_simulate_to_interruption

        (
            (interruptions, dynamics, initial_state, start_time, end_time),
            atemp_params,
            kwargs,
        ) = self._process_simulate_args_kwargs(msg)

        msg["value"] = diffeqdotjl_simulate_to_interruption(
            interruptions,
            dynamics,
            initial_state,
            start_time,
            end_time,
            atemp_params=atemp_params,
            **kwargs,
        )
        msg["done"] = True

    @staticmethod
    def _get_problem_shape(
        dynamics: PureDynamics[np.ndarray],
        initial_state: State[Tnsr],
        atemp_params: ATempParams[Tnsr],
    ) -> MappingShapePair:
        # Primarily due to interventions, but also different platings on the parameters, we may have a number of
        #  different shape requirements. We need to track separate problems that are compiled for each.
        initial_state = pre_broadcast_initial_state(
            dynamics, initial_state, atemp_params
        )
        return get_mapping_shape(initial_state), get_mapping_shape(atemp_params)

    # TODO g179du91 move to parent class as other solvers might also need to lazily compile?
    def _pyro__lazily_compile_problem(self, msg) -> None:

        (dynamics, initial_state, start_time, end_time), atemp_params, _ = (
            self._process_simulate_args_kwargs(msg)
        )

        initial_state = pre_broadcast_initial_state(
            dynamics, initial_state, atemp_params
        )
        initial_state_shape = get_mapping_shape(initial_state)
        atemp_params_shape = get_mapping_shape(atemp_params)
        problem_shape: MappingShapePair = (initial_state_shape, atemp_params_shape)

        # TODO this also should check to make sure that compilation kwargs are the same as the ones that were used
        #  to compile the solver? Or put those kwargs in the compilation mapping somewhere.
        # TODO eh we aren't really using compilation kwargs. They all go to solve.

        lazily_compiled_solver_by_shape = self._get_or_create_compilations_for_dynamics(
            dynamics
        )

        if problem_shape not in lazily_compiled_solver_by_shape:
            from chirho_diffeqpy.internals import diffeqdotjl_compile_problem

            lazily_compiled_solver_by_shape[problem_shape] = (
                diffeqdotjl_compile_problem(
                    dynamics,
                    initial_state,
                    start_time,
                    end_time,
                    atemp_params=atemp_params,
                )
            )

        msg["value"] = lazily_compiled_solver_by_shape[problem_shape]
        msg["done"] = True

    # TODO g179du91
    def _pyro__lazily_compile_event_fn_callback(self, msg) -> None:
        interruption, initial_state, atemp_params = msg[
            "args"
        ]  # type: Interruption, State[Tnsr], ATempParams[Tnsr]
        # TODO what if atemp_params passed as kwargs? Maybe fine b/c this is only used internally.

        # Not prebroadcasting here, as the event function will be called during the solve, which will be operating
        #  on the expanded state shape already.
        initial_state_shape = get_mapping_shape(initial_state)
        atemp_params_shape = get_mapping_shape(atemp_params)
        problem_shape: MappingShapePair = (initial_state_shape, atemp_params_shape)

        # TODO raise an error if any interruption.predicate are not ZeroEvent (which have an event_fn defined).
        lazily_compiled_event_fns_by_shape = (
            self._get_or_create_compilations_for_event_fns(
                interruption.predicate.event_fn
            )
        )

        if problem_shape not in lazily_compiled_event_fns_by_shape:
            from chirho_diffeqpy.internals import diffeqdotjl_compile_event_fn_callback

            lazily_compiled_event_fns_by_shape[problem_shape] = (
                diffeqdotjl_compile_event_fn_callback(
                    interruption, initial_state, atemp_params=atemp_params
                )
            )

        msg["value"] = lazily_compiled_event_fns_by_shape[problem_shape]
        msg["done"] = True

    def _pyro_check_dynamics(self, msg) -> None:
        from chirho_diffeqpy.internals import diffeqpy_check_dynamics

        (dynamics, initial_state, start_time, end_time), atemp_params, _ = (
            self._process_simulate_args_kwargs(msg)
        )

        diffeqpy_check_dynamics(
            dynamics, initial_state, start_time, end_time, atemp_params=atemp_params
        )
        msg["done"] = True
