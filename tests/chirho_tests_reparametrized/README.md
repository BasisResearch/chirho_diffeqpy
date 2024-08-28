# Reparametrizing Chirho Tests

This module provides a value and type dispatch system by which chirho fixtures can be converted to versions that are
compatible with the `DiffEqPy` solver handler.

`fixtures_imported_from_chirho.py` imports chirho fixtures that will be registered for reparametrization.
`global_reparametrization.py` and `per_test_reparametrization.py` provide value and type-dispatched conversions of those
fixtures into `DiffEqPy`-compatible versions.
The `reparametrization` module contains the dispatch functionality.
`mock_closure.py` provides a simple wrapper and `simulate` handling that allows for easier representation of torch
modules as pure dynamics functions.
