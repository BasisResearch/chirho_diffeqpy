import juliacall
import chirho
import sys
import os.path as osp

# <Force any `import tests` to resolve to chirho's tests folder>

chirho_root_path = osp.dirname(osp.dirname(chirho.__file__))

sys.path.remove(chirho_root_path)
sys.path.insert(0, chirho_root_path)
# </Force...>

# noinspection PyUnresolvedReferences
from tests.dynamical.dynamical_fixtures import (
    pure_sir_dynamics,
    SIRObservationMixin,
    SIRReparamObservationMixin,
    UnifiedFixtureDynamics,
    bayes_sir_model,
    sir_param_prior,
)
# noinspection PyUnresolvedReferences
from tests.dynamical.test_handler_composition import (
    UnifiedFixtureDynamicsReparam,
)
