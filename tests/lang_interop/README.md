# Language Interopation Testing

This directory contains tests for each language interoperation backend, primarily assessing forward and symbolic 
evaluation of python-defined functions from the julia side.

A suite of different python-defined functions are specified in `python_fixtures.py`, which are then evaluated from
julia functions defined in `julia_fixtures.jl`. Both forward and symbolic evaluation are tested.
