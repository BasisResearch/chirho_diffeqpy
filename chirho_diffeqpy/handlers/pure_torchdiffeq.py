from chirho.dynamical.handlers.solver import TorchDiffEq
from functools import partial
from copy import copy


class PureTorchDiffEq(TorchDiffEq):

    # noinspection PyFinal
    def _pyro_simulate(self, msg: dict) -> None:
        atemp_params = msg["kwargs"].pop("atemp_params", None)
        dynamics = msg["args"][0]
        if atemp_params is None:
            raise ValueError("PureTorchDiffEq requires atemp_params to be passed into simulate as a keyword argument.")

        closure_dynamics = partial(dynamics, atemp_params=atemp_params)
        msg["args"] = (closure_dynamics, *msg["args"][1:])

        return super()._pyro_simulate(msg)
