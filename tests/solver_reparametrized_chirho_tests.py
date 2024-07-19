import pytest
import juliacall  # Must precede even indirect torch imports to prevent segfault.
import chirho
import sys
import os.path as osp
from chirho_diffeqpy import DiffEqPy
from chirho_tests_reparametrized.fixtures_imported_from_chirho import chirho_root_path
from chirho.dynamical.handlers.solver import Solver
from chirho_tests_reparametrized.reparametrization import reparametrize_argument
import inspect
from typing import List, Tuple, Optional
import torch
from functools import wraps


def _args_include_solver(args: Tuple) -> bool:
    args = (args,) if not isinstance(args, tuple) else args
    return (
        # check for solver instances
        any(isinstance(arg, Solver) for arg in args)
        # check for solver types
        | any(inspect.isclass(arg) and issubclass(arg, Solver) for arg in args)
    )


def _marker_includes_solver(marker) -> bool:
    args_list = marker.args[1]
    return any(_args_include_solver(args) for args in args_list)


def _metafunc_includes_solver(metafunc) -> bool:
    markers = metafunc.definition.own_markers
    return any(_marker_includes_solver(marker) for marker in markers)


def _is_arg_group(args) -> bool:
    istuple = isinstance(args, tuple)
    return istuple


# TODO gnwg18fk may want to include arg_names here and have some kind of nested scope, or multi-scope lookup. Otherwise,
#  there is no way to specify different conversions when e.g. two different lambdas parametrize two different arguments
#  in the same test.
def _reparametrize_args(args: Tuple, test_id: Optional[str] = None) -> Optional[Tuple]:

    og_args_is_just_arg = not _is_arg_group(args)
    args = (args,) if og_args_is_just_arg else args

    new_args = []
    for arg in args:
        try:
            new_arg = reparametrize_argument(arg, scope=test_id)
        except NotImplementedError:
            # These are required to convert.
            # TODO maybe raise for a list of things and types that we require conversions for?
            if isinstance(arg, Solver):
                raise
            # TODO figure out a way to report whether a failed test had failed conversions? Or something to point
            #  the user to the fact that the test might have failed because the reparametrization failed.
            new_arg = arg  # just don't convert if not implemented.
        except Exception as e:
            raise NotImplementedError(
                f"Tried to reparametrize argument {arg} in test {test_id}, but failed with unexpected exception {e}."
            ) from e

        new_args.append(new_arg)

    return tuple(new_args) if not og_args_is_just_arg else new_args[0]


def _reparametrize_markers_in_place(metafunc) -> bool:

    markers = metafunc.definition.own_markers

    test_file, test_line, test_name = metafunc.definition.location
    test_id = metafunc.definition.nodeid

    # TODO First, identify if any of the markers
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
        # Also, TODO gnwg18fk.
        try:
            reparametrized_args_list = [_reparametrize_args(args, test_id=test_id) for args in args_list]
        except Exception as e:
            raise Exception(f"Errored while trying to reparametrize a solver argument in a parametrize marker for test "
                            f"{test_name} at {test_file}:{test_line} with parametrized arguments "
                            f"{args_name_str} = {args_list}. \n Original exception:\n {e}") from e

        args_list.clear()
        args_list.extend(reparametrized_args_list)


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

        if _metafunc_includes_solver(metafunc):
            _reparametrize_markers_in_place(metafunc)
            return

        # TODO Otherwise, skip this test.

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
# See also, in test_solver: FIXME hk0jd16g.
# Unused import to register the lang_interop machinery. This registers a bunch of type dispatch conversion functions.
from chirho_diffeqpy.lang_interop import julianumpy

# Also, import the global and per_test to register those dispatched reparametrizations.
import chirho_tests_reparametrized.global_reparametrizations
import chirho_tests_reparametrized.per_test_reparametrizations


# Programmatically execute chirho's dynamical systems test suite. Pass the plugin that will splice in the DiffEqPy
#  solver for testing.
retcode = pytest.main(
    [
        # TODO WIP expand to all dynamical tests.
        # f"{chirho_root_path}/tests/dynamical/test_log_trajectory.py",
        # f"{chirho_root_path}/tests/dynamical/test_solver.py",
        # f"{chirho_root_path}/tests/dynamical/test_noop_interruptions.py",
        f"{chirho_root_path}/tests/dynamical/test_static_observation.py",

        # The fault handler bottoms out for some reason related to juliacall and torch's weird segfaulting interaction.
        # The current implementation does NOT segfault, as long as juliacall is imported before torch, but adding
        #  the early julicall import causes some kind of permission error in the fault handler.
        # Solution: disable it.
        "-p", "no:faulthandler"
    ],
    plugins=[ReparametrizeWithDiffEqPySolver()]
)
sys.exit(retcode)
