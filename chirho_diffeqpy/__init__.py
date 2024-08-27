import juliacall
import torch
# Required by DiffEqPy.
torch.set_default_dtype(torch.float64)
from diffeqpy import de as _  # noop import to force diffeqpy shared julia environment activation.
from .handlers import DiffEqPy
from .internals import ATempParams, PureDynamics
