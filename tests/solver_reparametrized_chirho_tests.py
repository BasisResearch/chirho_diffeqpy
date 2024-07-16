import pytest
import juliacall  # Must precede even indirect torch imports to prevent segfault.
import chirho
import sys
import os.path as osp
# from chirho_diffeqpy import DiffEqPy
from chirho_tests_reparametrized.fixtures_imported_from_chirho import chirho_root_path
from chirho.dynamical.handlers.solver import Solver
from chirho_tests_reparametrized.reparametrizations import reparametrize_argument
import inspect
from typing import List, Tuple, Optional
import torch
from functools import wraps


def _reparametrize_args(args: Tuple, test_id: Optional[str] = None, arg_names: Optional[str] = None) -> Optional[Tuple]:

    # Look for any instances or subclasses of Solver in the args. If there is one, then this test should
    #  be reparametrized. TODO this won't recurse into collections that contain solvers.
    has_parametrized_solver = (
        # check for solver instances
        any(isinstance(arg, Solver) for arg in args)
        # check for solver types
        | any(inspect.isclass(arg) and issubclass(arg, Solver) for arg in args)
    )

    # If there is no reparametrization required, return None.
    if not has_parametrized_solver:
        return None

    # Otherwise, reparametrize all the args.
    new_args = []
    for arg in args:

        try:
            new_arg = reparametrize_argument(arg, test_id=test_id, arg_names=arg_names)
        except Exception as e:
            raise NotImplementedError(
                f"Found a solver to reparametrize, but could not reparametrize argument "
                f" {arg}. \n Original exception:\n {e}"
            ) from e

        new_args.append(new_arg)

    return tuple(new_args)


def _reparametrize_markers_in_place(metafunc) -> bool:

    markers = metafunc.definition.own_markers

    test_file, test_line, test_name = metafunc.definition.location
    test_id = metafunc.definition.nodeid

    reparametrized_any_marker = False

    for marker in markers:

        # We can only reparametrize args of parametrize markers.
        if marker.name != "parametrize":
            continue

        # TODO triple confirm that parametrize marks will never have more than 2 args here? And/or that the first
        #  and second args are always the arg name strings and list of arg values respectively?
        if not len(marker.args) > 1:
            continue
        # List of args tuples in this parametrize marker.
        args_list = marker.args[1]

        args_name_str: str = marker.args[0]

        # TODO processing the entire args_list isn't ideal, because if chirho ever introduces a second solver locally,
        #  then that solver will appear multiple times in the args_list, and be converted accordingly to a DiffEqPy
        #  solver. The pytest parametrizations though will have specified the same tests for different solvers, so here,
        #  this will result in redundant tests for all of those repeated cases where only the solver was varied.
        try:
            reparametrized_args_list = [_reparametrize_args(args, test_id=test_id, arg_names=args_name_str)
                                        for args in args_list]
        except NotImplementedError as e:
            raise NotImplementedError(f"Could not reparametrize a solver argument in a parametrize marker for test "
                                      f"{test_name} at {test_file}:{test_line} with parametrized arguments "
                                      f"{args_name_str} = {args_list}. \n Original exception:\n {e}") from e

        # Require that either all the args required reparametrization, or none of them did.
        was_reparametrized_per_arg = [p is not None for p in reparametrized_args_list]
        all_processed = all(was_reparametrized_per_arg)
        none_processed = not any(was_reparametrized_per_arg)

        if not (all_processed | none_processed):
            raise ValueError("Either all args in a parametrize marker must require reparametrization, or none of them."
                             " This can happen if some args involve a Solver instance or type while others do not."
                             f" Test {test_name} at {test_file}:{test_line} with parametrized arguments "
                             f"{args_name_str} = {args_list}.")

        if none_processed:
            continue

        args_list.clear()
        args_list.extend(reparametrized_args_list)

        reparametrized_any_marker = True

    return reparametrized_any_marker


# def _skip_wrapper(f):
#     @wraps(f)
#     def wrapper(*args, **kwargs):
#         return pytest.skip(
#             reason="No solver argument found to reparametrize. "
#                    "This test would not exercise the DiffEqPy solver backend."
#         )
#     return wrapper


class ReparametrizeWithDiffEqPySolver:
    @staticmethod
    def pytest_generate_tests(metafunc):

        # This logic reparametrizes any chirho tests that have 1) a solver parametrization that 2) includes the
        #  Solver type or instance as an argument. Then 3) reparametrizes the arguments accordingly to work with
        #  DiffEqPy and pure dynamics functions.
        reparametrized_any_marker = _reparametrize_markers_in_place(metafunc)

        if not reparametrized_any_marker:
            # Doesn't work, skips whole module.
            # pytest.skip("No solver parametrization marker found to reparametrize. This test would not exercise the"
            #             " DiffEqPy solver backend.")

            # Doesn't work, doesn't change anything.
            # metafunc.definition.add_marker(pytest.mark.skip(
            #     reason="No solver parametrization marker found to reparametrize. This test would not exercise the"
            #            " DiffEqPy solver backend."
            # ))

            # Doesn't work, doesn't change anything and skip never gets called.
            # metafunc.function = _skip_wrapper

            # Also doesn't work. Also tries to use an internal pytest
            # metafunc.definition.own_markers.clear()
            # skip_mark = pytest.Mark(name="skip", args=tuple(), kwargs=dict(
            #     reason="No solver parametrization marker found to reparametrize. This test would not exercise the"
            #            " DiffEqPy solver backend."
            # ))
            # metafunc.definition.own_markers.append(skip_mark)

            # TODO warnings module.
            print(f"WARNING: Would have skipped {metafunc.definition.nodeid}, as it wouldn't have exercised the DiffEqPy"
                  f"solver, but not yet figured out how to dynamically skip individual tests.")


# DiffEqPy requires float64s, so set the default here, and it will proc to the chirho tests.
torch.set_default_dtype(torch.float64)

# TODO what about generating separate solver instance parametrizations for every lang_interop backend?
# See also # FIXME hk0jd16g in test_solver.
# Unused import to register the lang_interop machinery.
from chirho_diffeqpy.lang_interop import julianumpy


# Programmatically execute chirho's dynamical systems test suite. Pass the plugin that will splice in the DiffEqPy
#  solver for testing.
retcode = pytest.main(
    [
        # TODO WIP expand to all dynamical tests.
        # f"{chirho_root_path}/tests/dynamical/test_log_trajectory.py",
        f"{chirho_root_path}/tests/dynamical/test_solver.py",

        # The fault handler bottoms out for some reason related to juliacall and torch's weird segfaulting interaction.
        # The current implementation does NOT segfault, as long as juliacall is imported before torch, but adding
        #  the early julicall import causes some kind of permission error in the fault handler.
        # Solution: disable it.
        "-p", "no:faulthandler"
    ],
    plugins=[ReparametrizeWithDiffEqPySolver()]
)
sys.exit(retcode)
