from solver_reparametrized_chirho_tests import main as _test_solver_and_return_exit_code


# Just a simple means of executing the reparametrized tests as one big test in this suite.
# If diagnosting test failure, run this with `pytest -s` to see the results of the reparametrized chirho tests,
#  or just run the reparametrized chirho tests directly by executing solver_reparametrized_chirho_tests.
def test_solver_on_chirho_suite():
    exit_code = _test_solver_and_return_exit_code()
    assert exit_code == 0
