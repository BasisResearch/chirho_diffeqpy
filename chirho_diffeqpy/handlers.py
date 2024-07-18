from __future__ import annotations

from typing import TypeVar, Dict, Tuple

import torch

from chirho.dynamical.internals.solver import Solver
from .internals import ATempParams, pre_broadcast, get_mapping_shape, MappingShape
from copy import copy

from diffeqpy import de

S = TypeVar("S")
T = TypeVar("T")

ATEMPPARAMS_KEY = "atemp_params"

MappingShapePair = Tuple[MappingShape, MappingShape]


class DiffEqPy(Solver[torch.Tensor]):

    def __init__(self):
        super().__init__()

        self.solve_kwargs = dict()
        self._lazily_compiled_solvers_by_shape: Dict[MappingShapePair, de.ODEProblem] = dict()
        self._dynamics_that_solver_was_compiled_with = None

    @staticmethod
    def _get_atemp_params_from_msg(msg) -> (ATempParams, Dict):
        # We don't modify the msg["kwargs"] in case some outer process is expecting it to remain constant.
        kwargs = copy(msg["kwargs"])
        # But the kwargs that we want to pass on to our stuff will need to have the params extracted,
        #  primarily for clarity.
        atemp_params = kwargs.pop(ATEMPPARAMS_KEY, None)
        if atemp_params is None:
            raise ValueError("DiffEqPy requires that dynamics are implemented as pure functions, which require"
                             " that parameters are passed into the simulate call alongside the dynamics function."
                             f" Add a '{ATEMPPARAMS_KEY}' key to the kwargs of the simulate call, and ensure your "
                             f" dynamics function accepts a '{ATEMPPARAMS_KEY}' argument after the state.")
        return atemp_params, kwargs

    def _process_simulate_args_kwargs(self, msg):
        atemp_params, kwargs = self._get_atemp_params_from_msg(msg)
        kwargs.update(self.solve_kwargs)
        return msg["args"], atemp_params, kwargs

    def _pyro_simulate_point(self, msg) -> None:
        from chirho_diffeqpy.internals import (
            diffeqdotjl_simulate_point,
        )

        (dynamics, initial_state, start_time, end_time), atemp_params, kwargs = self._process_simulate_args_kwargs(msg)

        msg["value"] = diffeqdotjl_simulate_point(
            dynamics, initial_state, start_time, end_time, atemp_params=atemp_params, **kwargs
        )
        msg["done"] = True

    def _pyro_simulate_trajectory(self, msg) -> None:
        from chirho_diffeqpy.internals import (
            diffeqdotjl_simulate_trajectory,
        )

        (dynamics, initial_state, timespan), atemp_params, kwargs = self._process_simulate_args_kwargs(msg)

        msg["value"] = diffeqdotjl_simulate_trajectory(
            dynamics, initial_state, timespan, atemp_params=atemp_params, **kwargs
        )
        msg["done"] = True

    def _pyro_simulate_to_interruption(self, msg) -> None:
        from chirho_diffeqpy.internals import (
            diffeqdotjl_simulate_to_interruption,
        )

        (interruptions, dynamics, initial_state, start_time, end_time), atemp_params, kwargs = (
            self._process_simulate_args_kwargs(msg)
        )

        msg["value"] = diffeqdotjl_simulate_to_interruption(
            interruptions,
            dynamics,
            initial_state,
            start_time,
            end_time,
            atemp_params=atemp_params,
            **kwargs
        )
        msg["done"] = True

    def _pyro_check_dynamics(self, msg) -> None:
        raise NotImplementedError

    # TODO g179du91 move to parent class as other solvers might also need to lazily compile?
    def _pyro__lazily_compile_problem(self, msg) -> None:

        (dynamics, initial_state, start_time, end_time), atemp_params, kwargs = self._process_simulate_args_kwargs(msg)

        # Primarily due to interventions, but also different platings on the parameters, we may have a number of
        #  different shape requirements. We need to track separate problems that are compiled for each.
        initial_state = pre_broadcast(dynamics, initial_state, atemp_params)
        initial_state_shape = get_mapping_shape(initial_state)
        atemp_params_shape = get_mapping_shape(atemp_params)
        problem_shape = (initial_state_shape, atemp_params_shape)

        if self._dynamics_that_solver_was_compiled_with is None:
            self._dynamics_that_solver_was_compiled_with = dynamics
        elif dynamics is not self._dynamics_that_solver_was_compiled_with:
            raise ValueError(
                "Lazily compiling a solver for a different dynamics than the one that was previously compiled."
                " This is implicitly asking for a recompilation of the underlying ODEProblem. Instead, create"
                f" a new {DiffEqPy.__name__} solver instance and simulate with the new dynamics in that context."
            )

        # TODO this also should check to make sure that compilation kwargs are the same as the ones that were used
        #  to compile the solver.

        if problem_shape not in self._lazily_compiled_solvers_by_shape:
            from chirho_diffeqpy.internals import diffeqdotjl_compile_problem

            self._lazily_compiled_solvers_by_shape[problem_shape] = diffeqdotjl_compile_problem(
                dynamics, initial_state, start_time, end_time, atemp_params=atemp_params, **kwargs
            )

        msg["value"] = self._lazily_compiled_solvers_by_shape[problem_shape]
        msg["done"] = True
