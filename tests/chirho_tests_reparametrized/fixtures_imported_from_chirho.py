import juliacall
import chirho
import os
import sys
import os.path as osp

# <Force any `import tests` to resolve to chirho's tests folder>

chirho_root_path = osp.dirname(osp.dirname(chirho.__file__))

sys.path.remove(chirho_root_path)
sys.path.insert(0, chirho_root_path)

# Check that chirho was installed as a source package with tests available.
chirho_root_dirs = set(os.listdir(chirho_root_path))
required_dirs = {"tests", "chirho", "docs"}
if not chirho_root_dirs.issuperset(required_dirs):
    raise ImportError(
        # FIXME 6fjj1ydg, change verbiage when resolved.
        f"Expected chirho root directory to contain {required_dirs}, got {chirho_root_dirs}. To run chirho "
        f" reparametrized chirho tests, we currently require a manual, local installation of chirho as a source package" 
        f" with tests available. See the readme for more information."
    )
# </Force...>

# noinspection PyUnresolvedReferences
from tests.dynamical.dynamical_fixtures import (
    # pure_sir_dynamics,  # not using b/c uses torch.sin, see # TODO md7291jdmd
    SIRObservationMixin,
    SIRReparamObservationMixin,
    UnifiedFixtureDynamics,
    UnifiedFixtureDynamicsBase,
    bayes_sir_model,
    sir_param_prior,
    build_event_fn_zero_after_tt
)
# noinspection PyUnresolvedReferences
from tests.dynamical.test_handler_composition import (
    UnifiedFixtureDynamicsReparam,
)
# noinspection PyUnresolvedReferences
from tests.dynamical.test_static_observation import (
    RandBetaUnifiedFixtureDynamics,
)
# noinspection PyUnresolvedReferences
from tests.dynamical.test_dynamic_interventions import (
    get_state_reached_event_f,
    model_with_param_in_state
)
