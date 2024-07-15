import pyro
from pyro.distributions import Uniform


def sir_dynamics(state, atemp_params):
    beta = atemp_params["beta"]
    gamma = atemp_params["gamma"]
    dS = -beta * state["S"] * state["I"]
    dI = beta * state["S"] * state["I"] - gamma * state["I"]
    dR = gamma * state["I"]
    return dict(S=dS, I=dI, R=dR)


def sample_sir_params():
    beta = pyro.sample("beta", Uniform(0, 1))
    gamma = pyro.sample("gamma", Uniform(0, 1))
    return dict(beta=beta, gamma=gamma)
