# Testing

This directory contains a set of repo-internal tests in addition to software that reparametrizes and runs chirho's
existing dynamical systems tests against the `DiffEqPy` solver handler.

`solver_reparametrized_chirho_tests.py` provides a `main` function that will reparametrize and execute chirho's
existing test suite.

`lang_interop` contains tests of julia-python language interoperation, primarily assessing forward and symbolic
evaluation of python-defined functions from the julia side.

`chirho_tests_reparametrized` is a module containing fixtures and tools for reparametrizing chirho's existing tests
to be compatible with the `DiffEqPy` solver handler.
