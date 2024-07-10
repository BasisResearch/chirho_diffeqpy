from __future__ import annotations

from typing import TypeVar

import torch

from chirho.dynamical.internals.solver import Solver

S = TypeVar("S")
T = TypeVar("T")


class DiffEqPy(Solver[torch.Tensor]):

    def __init__(self):
        super().__init__()

        self.solve_kwargs = dict()
        self._lazily_compiled_solver = None
        self._dynamics_that_solver_was_compiled_with = None

    # def __enter__(self):
    #     super().__enter__()
    #     # Clear on entrance so re-use of an instantiated solver handler
    #     #  doesn't result in unexpected behavior.
    #     self._lazily_compiled_solver = None

    def _pyro_simulate_point(self, msg) -> None:
        from chirho_diffeqpy.internals import (
            diffeqdotjl_simulate_point,
        )

        dynamics, initial_state_and_params, start_time, end_time = msg["args"]
        msg["kwargs"].update(self.solve_kwargs)

        msg["value"] = diffeqdotjl_simulate_point(
            dynamics, initial_state_and_params, start_time, end_time, **msg["kwargs"]
        )
        msg["done"] = True

    def _pyro_simulate_trajectory(self, msg) -> None:
        from chirho_diffeqpy.internals import (
            diffeqdotjl_simulate_trajectory,
        )

        dynamics, initial_state_and_params, timespan = msg["args"]
        msg["kwargs"].update(self.solve_kwargs)

        msg["value"] = diffeqdotjl_simulate_trajectory(
            dynamics, initial_state_and_params, timespan, **msg["kwargs"]
        )
        msg["done"] = True

    def _pyro_simulate_to_interruption(self, msg) -> None:
        from chirho_diffeqpy.internals import (
            diffeqdotjl_simulate_to_interruption,
        )

        interruptions, dynamics, initial_state_and_params, start_time, end_time = msg["args"]
        msg["kwargs"].update(self.solve_kwargs)
        msg["value"] = diffeqdotjl_simulate_to_interruption(
            interruptions,
            dynamics,
            initial_state_and_params,
            start_time,
            end_time,
            **msg["kwargs"],
        )
        msg["done"] = True

    def _pyro_check_dynamics(self, msg) -> None:
        raise NotImplementedError

    # TODO g179du91 move to parent class as other solvers might also need to lazily compile?
    def _pyro__lazily_compile_problem(self, msg) -> None:

        dynamics, initial_state_ao_params, start_time, end_time = msg["args"]

        if self._lazily_compiled_solver is None:
            from chirho_diffeqpy.internals import diffeqdotjl_compile_problem

            msg["kwargs"].update(self.solve_kwargs)

            self._lazily_compiled_solver = diffeqdotjl_compile_problem(
                dynamics, initial_state_ao_params, start_time, end_time, **msg["kwargs"]
            )
            self._dynamics_that_solver_was_compiled_with = dynamics
        elif dynamics is not self._dynamics_that_solver_was_compiled_with:
            raise ValueError(
                "Lazily compiling a solver for a different dynamics than the one that was previously compiled."
            )

        msg["value"] = self._lazily_compiled_solver
        msg["done"] = True
