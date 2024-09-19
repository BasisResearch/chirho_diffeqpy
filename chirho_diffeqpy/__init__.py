# must precede any torch import to prevent segfault
import juliacall  # noqa: F401
import torch

# Required by DiffEqPy.
torch.set_default_dtype(torch.float64)

from .load_julia_env import load_julia_env as _load_julia_env  # noqa: F401, E402

# This isn't strictly necessary, but loads all julia dependencies at once.
#  This makes for easier progress tracking during the precompilation process.
# This also must run before any diffeqpy imports, as it requires slightly
#  different versions and julia dependency exposure.
_load_julia_env()

from .handlers import DiffEqPy  # noqa: F401, E402
from .internals import ATempParams, PureDynamics  # noqa: F401, E402
