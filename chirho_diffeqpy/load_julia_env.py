from diffeqpy.__init__ import _ensure_julia_installed


# from diffeqpy.__init__ import load_julia_packages
# A rehash of the load_julia_packages function in diffeqpy that takes versions.
# Much is copied directly.
def load_and_pin_julia_packages(**names_versions):
    # To directly quote the diffeqpy code:
    # "This is terrifying to many people. However, it seems SciML takes pragmatic approach."
    _ensure_julia_installed()

    names = tuple(names_versions.keys())

    pkgspecs = [
        (
            f'Pkg.PackageSpec(name="{n}", version="{v}")'
            if v is not None
            else f'Pkg.PackageSpec(name="{n}")'
        )
        for n, v in names_versions.items()
    ]
    pinspecs = [
        f'Pkg.PackageSpec(name="{n}", version="{v}")'
        for n, v in names_versions.items()
        if v is not None
    ]

    script = f"""import Pkg
    Pkg.activate(\"diffeqpy\", shared=true)
    try
        import {", ".join(names)}
    catch e
        e isa ArgumentError || throw(e)
        Pkg.add([{", ".join(pkgspecs)}])
        Pkg.pin([{", ".join(pinspecs)}])
        Pkg.resolve()
        Pkg.precompile()
        import {", ".join(names)}
    end
    {", ".join(names)}"""

    # Unfortunately, `seval` doesn't support multi-line strings
    # https://github.com/JuliaPy/PythonCall.jl/issues/433
    script = script.replace("\n", ";")

    # Must be loaded after `_ensure_julia_installed()`
    from juliacall import Main as jl

    return dict(zip(names_versions.keys(), jl.seval(script))), jl


LOADED_PACKAGES = dict()
_LOADED_ENV_PTR = [None]


def load_julia_env():
    # The diffeqpy import handles just the first three packages, but we also need access to the latter
    #  three dependencies.
    # Note that load_julia_packages also activates a Pkg environment with this stuff in it.

    if _LOADED_ENV_PTR[0] is None:
        loaded_packages, jl = load_and_pin_julia_packages(
            # Pinned to 7.11.0, as later versions introduce breaking changes. FIXME resolve
            DifferentialEquations="7.11.0",
            ModelingToolkit=None,
            PythonCall=None,
            Symbolics=None,
            ForwardDiff=None,
            SymbolicUtils=None,
            LinearSolve="2.22.1",
            BandedMatrices="1.7.3",
            # PDE stuff.
            OrdinaryDiffEq=None,
            MethodOfLines=None,
            DomainSets=None
        )

        LOADED_PACKAGES.clear()
        LOADED_PACKAGES.update(loaded_packages)
        _LOADED_ENV_PTR[0] = jl

    return _LOADED_ENV_PTR[0]
