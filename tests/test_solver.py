import logging
import time
from importlib import import_module

# Must precede even indirect torch imports to prevent segfault.
import juliacall  # noqa: F401
import numpy as np
import pyro
import pytest
import torch
from chirho.dynamical.handlers import DynamicIntervention, LogTrajectory
from chirho.dynamical.ops import State, simulate
from diffeqpy import de
from fixtures import logistic_growth_dynamics

from chirho_diffeqpy import ATempParams, DiffEqPy

pyro.settings.set(module_local_params=True)

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("solver", [DiffEqPy])
@pytest.mark.parametrize(
    "alg", [None, de.Vern7(), de.AutoTsit5(de.Rosenbrock23()), de.Tsit5(), "dont pass"]
)
@pytest.mark.parametrize(
    "lang_interop_backend", ["chirho_diffeqpy.lang_interop.julianumpy"]
)
def test_forward_correct(solver, alg, lang_interop_backend):

    # This loads the conversion operation overloads for a particular backend.
    # FIXME hk0jd16g will these overloads bleed into other tests?
    import_module(lang_interop_backend)

    u0: State = dict(x=torch.tensor(10.0).double())
    atemp_params: ATempParams = dict(c=torch.tensor(0.1).double())
    timespan = torch.linspace(1.0, 10.0, 10).double()

    def dynamics(s: State, p: ATempParams) -> State:
        return dict(x=-(s["x"] * p["c"]))

    solver_instance = solver(alg=alg) if alg != "dont pass" else solver()

    with LogTrajectory(timespan) as lt:
        with solver_instance:
            simulate(
                dynamics,
                u0,
                timespan[0] - 1.0,
                timespan[-1] + 1.0,
                atemp_params=atemp_params,
            )

    correct = torch.tensor(
        [
            9.0484,
            8.1873,
            7.4082,
            6.7032,
            6.0653,
            5.4881,
            4.9659,
            4.4933,
            4.0657,
            3.6788,
        ],
        dtype=torch.float64,
    )

    assert torch.allclose(lt.trajectory["x"], correct, atol=1e-4)


dynfuncs = [
    # Unsimplified defs just ensure the solutions are stable but still exercise the relevant ops. Julia does compile
    #  these away, but not until it's able to successfully push symbolics through the unsimplified python func.
    lambda s, p: -s["x"] * p["c"],
    lambda s, p: -(s["x"] * p["c"]),
    lambda s, p: -0.5 * s["x"] * p["c"],  # w/ python numeric type
    lambda s, p: -s["x"] / (1.0 / p["c"]),
    lambda s, p: -s["x"] * (p["c"] + p["c"] - p["c"]),
    lambda s, p: -s["x"] ** (p["c"] / p["c"]),
    # The last three dynfuncs test numpy ufunc dispatch to julia.
    # .ravel and slice ensure that the return here always matches the initial state (the 2x2 matmul gives you 4 elems).
    lambda s, p: -((np.atleast_2d(s["x"]).T @ np.atleast_2d(p["c"])) * s["x"]).ravel()[
        : s["x"].size
    ],
    lambda s, p: -(
        np.matmul(np.atleast_2d(s["x"]).T, np.atleast_2d(p["c"])) * s["x"]
    ).ravel()[: s["x"].size],
    lambda s, p: np.sin(s["t"])
    + np.sin(np.pi + s["t"])
    - s["x"] * np.exp(np.log(p["c"])),
]


@pytest.mark.parametrize("solver", [DiffEqPy])
@pytest.mark.parametrize("x0", [torch.tensor(10.0), torch.tensor([10.0, 5.0])])
@pytest.mark.parametrize("c_", [torch.tensor(0.5), torch.tensor([0.5, 0.3])])
@pytest.mark.parametrize("dynfunc", dynfuncs)
@pytest.mark.parametrize(
    "lang_interop_backend", ["chirho_diffeqpy.lang_interop.julianumpy"]
)
def test_compile_forward_and_gradcheck(solver, x0, c_, dynfunc, lang_interop_backend):

    # This loads the conversion operation overloads for a particular backend.
    # FIXME hk0jd16g will these overloads bleed into other tests?
    import_module(lang_interop_backend)

    c_ = c_.double().requires_grad_()
    timespan = torch.linspace(1.0, 10.0, 10).double()

    # TODO gradcheck wrt time.

    def dynamics(s: State, p: ATempParams) -> State:
        dx = dynfunc(s, p)

        # for the test case where there are two parameters but only one (scalar) state variable.
        if x0.ndim == 0 and c_.ndim > 0:
            dx = dx.sum()

        return dict(x=dx)

    def wrapped_simulate(c):  # TODO test diff wrt time and initial state?
        u0: State = dict(x=x0.double())
        atemp_params: ATempParams = dict(c=c.double())
        with LogTrajectory(timespan) as lt:
            simulate(
                dynamics,
                u0,
                timespan[0] - 1.0,
                timespan[-1] + 1.0,
                atemp_params=atemp_params,
            )
        return lt.trajectory["x"]

    # This goes outside the gradcheck b/c DiffEqDotJl lazily compiles the problem.
    with solver():
        torch.autograd.gradcheck(
            wrapped_simulate, c_, atol=1e-4, check_undefined_grad=True
        )


@pytest.mark.parametrize("solver", [DiffEqPy])
@pytest.mark.parametrize(
    "lang_interop_backend", ["chirho_diffeqpy.lang_interop.julianumpy"]
)
def test_second_run_of_double_dynamical_interventions(solver, lang_interop_backend):
    """
    This tests for a now-resolved bug where the second run of a simulation with two dynamical interventions would
    fail. This was likely due to some kind of caching in the VectorContinuousCallback that was being cached. Now
    we cache only the inner compilation of the event_fn, and re-wrap in new callbacks for each simulation.
    """

    # This loads the conversion operation overloads for a particular backend.
    # FIXME hk0jd16g will these overloads bleed into other tests?
    import_module(lang_interop_backend)

    solver_instance = solver()

    def growth_event_fn_x(t, state, atemp_params):
        return state["x"] - 3.0

    def growth_event_fn_y(t, state, atemp_params):
        return state["y"] - 5.0

    times = []
    results = []
    for _ in range(3):
        start_time = time.time()
        with solver_instance, LogTrajectory(torch.linspace(0.0, 30.0, 30)) as lt:
            with DynamicIntervention(growth_event_fn_x, dict(x=torch.tensor(1.0))):
                with DynamicIntervention(growth_event_fn_y, dict(y=torch.tensor(1.0))):
                    simulate(
                        logistic_growth_dynamics,
                        dict(x=torch.tensor(1.0), y=torch.tensor(1.0)),
                        torch.tensor(0.0),
                        torch.tensor(30.0),
                        atemp_params=dict(a=torch.tensor(0.5), b=torch.tensor(10.0)),
                    )
        times.append(time.time() - start_time)
        results.append(lt.trajectory)

    # Assert that the first time is 10x longer than the subsequent runs (for compilation).
    # In practice, it's much more than this.
    assert times[0] > 10 * times[1]
    assert times[0] > 10 * times[2]

    # Assert that the trajectories are all the same.
    assert torch.allclose(results[0]["x"], results[1]["x"])
    assert torch.allclose(results[0]["x"], results[2]["x"])
    assert torch.allclose(results[0]["y"], results[1]["y"])
    assert torch.allclose(results[0]["y"], results[2]["y"])


def test_compiling_with_different_shapes():
    # 1. Sample from two different platings from the prior, and make those plating show up in
    #      solver._lazily_compiled_solvers_by_shape
    pytest.skip("Not Implemented")


# TODO test that the informative error fires when dstate returns something that isn't the right shape.
# TODO test that simulating with a dynamics different from that which a solver was compiled for fails.
# TODO test whether float64 is properly required of initial state for diffeqpy backend.
# TODO test whether float64 is properly required of returned dstate even if initial state passes float64 check.
