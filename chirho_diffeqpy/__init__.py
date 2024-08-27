import juliacall  # must precede all any torch import to prevent segfault
import torch
# Required by DiffEqPy.
torch.set_default_dtype(torch.float64)

from .handlers import DiffEqPy
from .internals import ATempParams, PureDynamics
