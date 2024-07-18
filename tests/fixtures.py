# ty to ChatGPT
import numpy as np
from contextlib import nullcontext
import pyro
import pyro.distributions as dist


def flux_dynamics(state, atemp_params):
    # Example parameters
    alpha = atemp_params["a"]
    beta = atemp_params["b"]

    # State variables
    x = state["x"]
    y = state["y"]

    # Flux equations
    dx = alpha * x * (1. - x / 10.) - beta * x * y
    dy = beta * x * y - y

    return dict(x=dx, y=dy)


def decay_dynamics(state, atemp_params):
    # Example parameters
    decay_rate_a = atemp_params["a"]
    decay_rate_b = atemp_params["b"]

    # State variables
    x = state["x"]
    y = state["y"]
    t = state["t"]

    # Decay equations
    dx = -decay_rate_a * (np.sin(t) + 2.) * x
    dy = -decay_rate_b * y

    return dict(x=dx, y=dy)


def predator_prey_dynamics(state, atemp_params):
    # Example parameters
    alpha = atemp_params["a"]
    beta = atemp_params["b"]
    delta = atemp_params["a"]
    gamma = atemp_params["b"]

    # State variables
    x = state["x"]
    y = state["y"]

    # Predator-Prey equations
    dx = alpha * x - beta * x * y
    dy = delta * x * y - gamma * y

    return dict(x=dx, y=dy)


def chemical_reaction_dynamics(state, atemp_params):
    # Example parameters
    k1 = atemp_params["a"]
    k2 = atemp_params["b"]

    # State variables
    x = state["x"]
    y = state["y"]

    # Chemical reaction equations
    dx = -k1 * x + k2 * y
    dy = k1 * x - k2 * y

    return dict(x=dx, y=dy)


def logistic_growth_dynamics(state, atemp_params):
    # Example parameters
    r = atemp_params["a"]
    K = atemp_params["b"]

    # State variables
    x = state["x"]
    y = state["x"]

    # Logistic growth equation
    dx = r * x * (1 - x / K)
    dy = r * y * (1 - y / K)

    return dict(x=dx, y=dy)


# TODO a flux model that uses matmul


# List of function references
ab_xy_dynfuncs = [
    flux_dynamics,
    decay_dynamics,
    predator_prey_dynamics,
    chemical_reaction_dynamics,
    logistic_growth_dynamics,
]


def ab_xy_prior(
    xplatesize=1,
    yplatesize=1,
    aplatesize=1,
    bplatesize=1,
):
    xplate = pyro.plate("xplate", xplatesize, dim=-4) if xplatesize is not None else nullcontext()
    yplate = pyro.plate("yplate", yplatesize, dim=-3) if yplatesize is not None else nullcontext()
    aplate = pyro.plate("aplate", aplatesize, dim=-2) if aplatesize is not None else nullcontext()
    bplate = pyro.plate("bplate", bplatesize, dim=-1) if bplatesize is not None else nullcontext()

    with xplate:
        x = pyro.sample("x", dist.Uniform(0.1, 1))
        with yplate:
            y = pyro.sample("y", dist.Uniform(0.1, 1))

    with aplate:
        a = pyro.sample("a", dist.Uniform(0.1, 1))
        with bplate:
            b = pyro.sample("b", dist.Uniform(0.1, 1))

    return dict(x=x, y=y), dict(a=a, b=b)
