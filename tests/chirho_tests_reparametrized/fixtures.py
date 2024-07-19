import numpy as np


def isalambda(f):
    return callable(f) and f.__name__ == "<lambda>"


# Exactly as written in the original chirho fixture, but uses np.sin instead of torch.sin....
# TODO md7291jdmd dispatch sin on type in the original chirho code?
def pure_sir_dynamics(state, atemp_params):
    beta = atemp_params["beta"]
    gamma = atemp_params["gamma"]

    dX = dict()

    beta = beta * (
            1.0 + 0.1 * np.sin(0.1 * state["t"])
    )  # beta oscilates slowly in time.

    dX["S"] = -beta * state["S"] * state["I"]  # noqa
    dX["I"] = beta * state["S"] * state["I"] - gamma * state["I"]  # noqa
    dX["R"] = gamma * state["I"]  # noqa

    return dX
