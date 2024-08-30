from typing import TypeAlias, Union

import numpy as np
import pyro
import torch
from chirho.dynamical.ops import State

from .solver import ATempParams, DiffEqPy

_ArrayLike: TypeAlias = Union[torch.Tensor, np.ndarray]


class MockDynamicsClosureMixin(pyro.nn.PyroModule):
    """
    Used in tandem with DiffEqPyMockClosureCapable to enable closure-style dispatch of simulate.
    """

    @property
    def atemp_params(self):
        raise NotImplementedError()

    def forward(
        self, state: State[_ArrayLike], atemp_params: ATempParams[_ArrayLike]
    ) -> State[_ArrayLike]:
        raise NotImplementedError(
            "This method should be overriden by subclasses or mixins."
        )


class DiffEqPyMockClosureCapable(DiffEqPy):
    """
    This enables closure-style dispatch of simulate, which allows us to operate with the same dynamics/parameter
    coupling that the chirho tests use by defining dynamics as torch modules.
    """

    # noinspection PyMethodMayBeStatic,PyFinal
    def _pyro_simulate(self, msg: dict) -> None:
        dynamics = msg["args"][0]

        if isinstance(dynamics, MockDynamicsClosureMixin):
            if "atemp_params" in msg["kwargs"]:
                raise ValueError(
                    "MockDynamicsClosureAbstract instances should not have atemp_params passed in "
                    "to simulate, but rather exposed via the overriden atemp_params property."
                )
            msg["kwargs"]["atemp_params"] = dynamics.atemp_params
            return super()._pyro_simulate(msg)
