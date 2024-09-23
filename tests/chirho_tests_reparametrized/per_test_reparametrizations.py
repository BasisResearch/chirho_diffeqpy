from .fixtures import MockClosureDynamicsDirectPass, isalambda
from .reparametrization import reparametrize_argument_by_type

test_log_trajectory__test_start_end_time_collisions = (
    "tests/dynamical/test_log_trajectory.py::test_start_end_time_collisions"
)


# This test uses a lambda for dynamics with no parameters.
@reparametrize_argument_by_type.register(
    type(lambda: 0), scope=test_log_trajectory__test_start_end_time_collisions
)
def _(f, *args, **kwargs):
    assert isalambda(f)
    return MockClosureDynamicsDirectPass(
        dynamics=lambda s, _: f(s), atemp_params=dict()
    )
