The file `solver_reparametrized_chirho_tests.py` will not run under normal pytest conditions. It cannot, in fact,
without a number of issues occurring. It must instead be run as a normal python script that will invoke chirho's
dynamical systems module tests, but with the DiffEqPy solver instead of the chirho-provided solvers.
