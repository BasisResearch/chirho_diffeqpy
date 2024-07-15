import juliacall
import chirho
import sys
import os.path as osp

chirho_root_path = osp.dirname(osp.dirname(chirho.__file__))

# Going to get the UnifiedFixtureDynamics from chirho, so we can use it to know when the reparametrized dynamics
#  should be consistent with the SIR encoded in that model type.
chirho_dynamical_tests_path = osp.join(chirho_root_path, "tests/dynamical")

if chirho_dynamical_tests_path not in sys.path:
    sys.path.insert(0, chirho_dynamical_tests_path)

# noinspection PyUnresolvedReferences
from dynamical_fixtures import UnifiedFixtureDynamics

# TODO import any other fixtures that might be useful
