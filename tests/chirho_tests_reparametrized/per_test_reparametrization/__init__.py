from .reparam_test_log_trajectory import *
from .reparam_test_solver import *

# TODO I don't like this set up...hmmm.
#  A lot of this can go away, I think, with a value-dispatch reparameterization scheme. Then we can import
#  chirho fixtures and dispatch on them exactly.


per_test_reparametrization = {
    "tests/dynamical/test_log_trajectory.py::test_start_end_time_collisions":
        test_log_trajectory__test_start_end_time_collisions,
    'tests/dynamical/test_solver.py::test_broadcasting':
        test_solver__test_broadcasting,
}
