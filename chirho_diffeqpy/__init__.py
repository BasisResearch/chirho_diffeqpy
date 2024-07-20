import juliacall
import torch
# Required by DiffEqPy.
torch.set_default_dtype(torch.float64)
from .handlers import DiffEqPy
from .internals import ATempParams, PureDynamics
