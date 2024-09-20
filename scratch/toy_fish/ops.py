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


# Effectful virtual fish function for easy hot-swapping.
class NoVirtualFish(NotImplementedError):
    pass


# noinspection PyProtectedMember
@pyro.poutine.runtime.effectful(type="virtual_fish_dynamics")
def virtual_fish_dynamics(state, atemp_params):
    raise NoVirtualFish()


@singledispatch
def sin(x):
    return np.sin(x)


@sin.register(torch.Tensor)
def _(x):
    return torch.sin(x)


@singledispatch
def cos(x):
    return np.cos(x)


@cos.register(torch.Tensor)
def _(x):
    return torch.cos(x)