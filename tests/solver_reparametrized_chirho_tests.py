import pytest
import juliacall  # Must precede even indirect torch imports to prevent segfault.
import chirho
import sys
import os.path as osp
# from chirho_diffeqpy import DiffEqPy
from chirho_tests_reparametrized.fixtures_from_chirho import chirho_root_path
from chirho.dynamical.handlers.solver import Solver
from chirho_tests_reparametrized.reparametrizations import reparametrize_argument
import inspect


class ReparametrizeWithDiffEqPySolver:
    @staticmethod
    def pytest_generate_tests(metafunc):

        found_solver_to_reprametrize = False

        # This logic reparametrizes any chirho tests that have 1) a solver parametrization that 2) includes the
        #  TorchDiffEq solver as an argument (TODO but not currently a keyword argument).
        #   Then 3) replace the entire solver parametrization with just [DiffEqPy]. Note that we can fully replace with
        #   just DiffEqPy because we only want to test this repo's solver. Chirho tests its own solvers.
        for marker in metafunc.definition.own_markers:

            if marker.name != "parametrize":
                continue

            if not len(marker.args) > 1:
                continue

            # FIXME this isn't ideal, because if chirho ever introduces a second solver locally, then this
            #  will generate redundant tests for all the cases that test the second solver (it will just test
            #  DiffEqPy instead a second time).
            # TODO triple confirm that parametrize marks will never have more than 2 args here? And/or that the first
            #  and second args are always the arg name strings and list of arg values respectively?
            for arg_vals in marker.args[1]:

                has_parametrized_solver = (
                    any(isinstance(arg, Solver) for arg in arg_vals)
                    | any(inspect.isclass(arg) and issubclass(arg, Solver) for arg in arg_vals)
                )

                if not has_parametrized_solver:
                    continue

                # TODO this should be true for all of them...but we probably want to handle the 1:M nuance here.
                found_solver_to_reprametrize = True

                new_arg_vals = []
                for arg in arg_vals:
                    try:
                        new_arg_vals.append(reparametrize_argument(arg))
                    except NotImplementedError as e:
                        raise NotImplementedError(
                            f"Found a solver to reparametrize, but could not reparametrize argument "
                            f" {arg} in {metafunc.definition.nodeid}."
                        )

                # Not sure if doing this in-place is necessary.
                # FIXME this won't work b/c arg_vals is a tuple of arguments in a larger list.
                #  Instead, we need to append the converted new_arg_vals to an outer list, and then perform this
                #  logic on the outer list.
                raise NotImplementedError("FIXME see comment immediately above.")
                arg_vals.clear()
                arg_vals.extend(new_arg_vals)

        if not found_solver_to_reprametrize:
            metafunc.definition.own_markers.append(
                pytest.mark.skip(reason="No solver parametrization found to reparametrize.")
            )


[print(p) for p in sys.path]

# Programmatically execute chirho's dynamical systems test suite. Pass the plugin that will splice in the DiffEqPy
#  solver for testing.
retcode = pytest.main(
    [
        # TODO WIP expand to all dynamical tests.
        "-x", f"{chirho_root_path}/tests/dynamical/test_log_trajectory.py",

        # The fault handler bottoms out for some reason related to juliacall and torch's weird segfaulting interaction.
        # The current implementation does NOT segfault, as long as juliacall is imported before torch, but adding
        #  the early julicall import causes some kind of permission error in the fault handler.
        # Solution: disable it.
        "-p", "no:faulthandler"
    ],
    plugins=[ReparametrizeWithDiffEqPySolver()]
)
sys.exit(retcode)
