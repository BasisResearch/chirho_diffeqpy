from chirho_diffeqpy import DiffEqPy, ATempParams
from chirho.dynamical.handlers.solver import TorchDiffEq
from chirho.dynamical.ops import simulate
from chirho.dynamical.handlers.trajectory import LogTrajectory
from chirho.dynamical.handlers import StaticBatchObservation, StaticIntervention, DynamicIntervention
from chirho.counterfactual.handlers import MultiWorldCounterfactual
from chirho.observational.handlers import condition
from chirho.dynamical.ops import State
import numpy as np
import torch
from pyro import sample, set_rng_seed
from pyro.distributions import Uniform, Poisson
from typing import Tuple, Optional, Union, Callable
from functools import partial
from numbers import Real
import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDelta, AutoMultivariateNormal
from contextlib import nullcontext
# The DiffEqPy backend interfaces with julia, so we must load a julia<>python interop backend.
import chirho_diffeqpy.lang_interop.julianumpy
from contextlib import ExitStack, nullcontext
from pyro.contrib.autoname import scope
# Global params are weird.
from functools import singledispatch, partial
import pyro
import torch
import zuko
from pyro.contrib.zuko import ZukoToPyro
from pyro.contrib.easyguide import easy_guide
from collections import namedtuple
from chirho.observational.handlers import condition
from itertools import product


class VirtualFishContext(pyro.poutine.messenger.Messenger):
    def __init__(self, virtual_fish_dynamics_fn, initial_state, atemp_params):
        self.virtual_fish_dynamics_fn = virtual_fish_dynamics_fn
        self.initial_state = initial_state
        self.atemp_params = atemp_params
        super().__init__()

    def _pyro_virtual_fish_dynamics(self, msg: dict) -> None:
        msg["value"] = self.virtual_fish_dynamics_fn(*msg["args"], **msg["kwargs"])
        assert 'vfish_theta' in msg["value"]
        msg["done"] = True

    def _pyro_simulate(self, msg: dict) -> None:
        # Add the virtual fish parameters and starting state that are associated with the passed trajectory function.
        og_atemp_params = msg["kwargs"].pop("atemp_params")
        og_initial_state = msg["args"][1]

        atemp_params = dict(**og_atemp_params, **self.atemp_params)
        initial_state = dict(**og_initial_state, **self.initial_state)

        assert 'vfish_theta' in initial_state

        msg["kwargs"]["atemp_params"] = atemp_params
        msg["args"] = (msg["args"][0], initial_state, *msg["args"][2:])


# TorchDiffEq backend interoperation.
class PureTorchDiffEq(TorchDiffEq):

    # noinspection PyFinal
    def _pyro_simulate(self, msg: dict) -> None:
        atemp_params = msg["kwargs"].pop("atemp_params", None)
        dynamics = msg["args"][0]
        if atemp_params is None:
            raise ValueError("PureTorchDiffEq requires atemp_params to be passed into simulate as a keyword argument.")

        print("params", atemp_params)
        print("u0", msg["args"][1])

        closure_dynamics = partial(dynamics, atemp_params=atemp_params)
        msg["args"] = (closure_dynamics, *msg["args"][1:])

        return super()._pyro_simulate(msg)
