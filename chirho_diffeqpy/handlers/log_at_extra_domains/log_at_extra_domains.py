from pyro.poutine.messenger import Messenger
import torch
from ...load_julia_env import load_julia_env
from os.path import join, abspath, dirname

jl = load_julia_env()

local_dir = dirname(abspath(__file__))
with open(join(local_dir, "flatten_multi_domain_interpolation.jl"), "r") as f:
    flatten_multi_domain_interpolation_code = f.read()

flatten_multi_domain_interpolation = jl.seval(flatten_multi_domain_interpolation_code)


class LogAtExtraDomains(Messenger):
    """
    A handler that allows a user to log the solution at additional domains. This is primarily useful when
    simulating partial differential equations of both time and some other domain (usually space).
    The interpolated solution will not be differentiable wrt these logging points, so they cannot require grad.
    """
    def __init__(self, *logging_points: torch.tensor):
        super().__init__()

        # Check loggoing points.
        for p in logging_points:
            if p.requires_grad:
                raise ValueError(
                    "Logging points must not require grad. The computation graph will not be constructed"
                    " for the interpolated solution at these points, so gradients will not be available."
                )

            if p.ndim != 1:
                raise ValueError("Logging points must be 1D tensors.")

        # Convert the tensors all to numpy arrays.
        self._logging_points = [point.detach().numpy() for point in logging_points]

    def _pyro_interpolate_solution(self, msg):
        sol, tspan = msg["args"]
        interpolation = sol(tspan, *self._logging_points)

        # Where K is the number of solution functions in the system (e.g. u, v, w for a PDE).
        # .shape == (T, K * prod(len(lp) for lp in self._logging_points))
        flat_interpolation = flatten_multi_domain_interpolation(interpolation)

        msg["value"] = flat_interpolation
        msg["done"] = True
