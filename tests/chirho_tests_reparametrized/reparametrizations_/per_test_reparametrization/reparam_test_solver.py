from ..fixtures import sir_dynamics, sample_sir_params

# TODO can we make these also single dispatch mechanisms that are only loaded for specific tests?
# TODO cz right now these will need to handle all their own dispatching via switch statements etc.


def test_solver__test_broadcasting(
        dynamics_kwargs_builder,
        arg_names: str = None
):
    # TODO this shouldn't be the full list of arg names, and should instead e.g. just be the second arg listed below.
    if arg_names != "backend, dynamics__simulate_kwargs__builder":
        raise NotImplementedError("This reparam only works when args are 'backend, dynamics__simulate_kwargs__builder'.")

    def build_dynamics_and_simulation_kwargs():
        return sir_dynamics, dict(atemp_params=sample_sir_params())

    return build_dynamics_and_simulation_kwargs

