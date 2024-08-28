# chirho_diffeqpy
An experimental diffeqpy (DifferentialEquations.jl) backend for chirho's dynamical systems module.

See `docs/source/performance_comparison.ipynb` for a comparison between chirho's default `TorchDiffEq` backend and the
`DiffEqPy` backend provided here.


## Installation

`pip install git+https://github.com/BasisResearch/chirho_diffeqpy.git`

Note that on the first import of the package, a large number of julia packages will be
installed and precompiled in a shared environment called "diffeqpy". This can take some time (up to 30 minutes).

### Installing with Tests

[//]: # (Remove after resolving FIXME 6fjj1ydg:)
[//]: # ( `tests/chirho_tests_reparametrized/fixtures_imported_from_chirho`)
[//]: # ( `.github/workflows/test.yml`)

The module comes with some internal tests, but primarily relies on a reparametrization of chirho's test suite to assess
correctness. In order to run these tests, we currently require that chirho is manually installed as a local source 
distribution before running the tests. To run tests locally:

[//]: # (In step 2, TODO pull version from requirements.txt)
[//]: # (Also, this will install and then reinstall chirho, but the manual must follow so that it takes precedence.)

1. create a virtual environment
2. clone this repo: `git clone https://github.com/BasisResearch/chirho_diffeqpy.git`
3. install from source: `pip install -e './chirho_diffeqpy[test]'`
4. clone chirho: `git clone https://github.com/BasisResearch/chirho.git; cd chirho; git reset --hard cda2f56; cd ..`
5. install chirho from source: `pip install -e './chirho[dynamical,test]'`


Finally, tests can be run like so:
`pytest chirho_diffeqpy/tests`