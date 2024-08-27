from diffeqpy.__init__ import load_julia_packages
from juliacall import Main as jl


def load_julia_env():
    # The diffeqpy import handles just the first three packages, but we also need access to the latter two dependencies.
    # Use this instead of importing Main directly to ensure the proper environment is loaded.
    # Note that load_julia_packages also activates a Pkg environment with this stuff in it.

    # Running this first, as it activates the diffeqpy package environment.
    load_julia_packages(
        # "DifferentialEquations",  # <-- w/o this for now. See below.
        "ModelingToolkit",
        "PythonCall",
        "Symbolics",
        "ForwardDiff",
        "SymbolicUtils"
    )

    # <FIXME Pin to 7.11.0 diffeq>
    # Later versions have breaking changes that need addressing...
    pkg = "DifferentialEquations"
    v = "7.11.0"
    pkg_spec = f'Pkg.PackageSpec(name="{pkg}", version="{v}")'

    # Manually do what the load_julia_packages does, but such that we can specify a version explicitly.
    script = f'''
    import Pkg
    try
        import {pkg}
    catch e
        e isa ArgumentError || throw(e)
        Pkg.add({pkg_spec})
        Pkg.pin({pkg_spec})
        import {pkg}
    end
    '''
    # script = f'''
    # Pkg.add({pkg_spec})
    # Pkg.pin({pkg_spec})
    # import {pkg}
    # '''
    jl.seval(script)

    # ...and pin it so it doesn't update on later Pkg.add calls.
    # </FIXME Pin to 7.11.0 diffeq>

    return jl
