from diffeqpy.__init__ import load_julia_packages
from juliacall import Main


def load_julia_env():
    # The diffeqpy import handles just the first three packages, but we also need access to the latter two dependencies.
    # Use this instead of importing Main directly to ensure the proper environment is loaded.
    # Note that load_julia_packages also activates a Pkg environment with this stuff in it.

    load_julia_packages("DifferentialEquations", "ModelingToolkit", "PythonCall", "Symbolics", "ForwardDiff")

    return Main
