# TODO can we make these also single dispatch mechanisms that are only loaded for specific tests?
# TODO cz right now these will need to handle all their own dispatching via e.g. switch statements etc.

def test_log_trajectory__test_start_end_time_collisions(
        lambda_dynamics,
        empty_simulation_kwargs,
        arg_names: str = None
):
    # The lambda dynamics don't take any parameters, but we still need to add the empty parameter dictionary.
    # Arguably, we could default to this behavior, but the current design requires that the user explicitly specify
    #  that there are indeed no parameters.
    empty_simulation_kwargs["atemp_params"] = dict()
    return lambda_dynamics, empty_simulation_kwargs
