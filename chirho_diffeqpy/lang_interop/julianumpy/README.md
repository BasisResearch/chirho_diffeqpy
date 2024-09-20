# Evaluating `numpy` from Julia

This backend allows julia code to evaluate python functions defined to support numpy operations and array manipulation.
It takes a brute-force approach by converting `juliacall` arrays to numpy arrays of wrapped julia scalars. This allows
for the straightforward interpretation of numpy broadcasting and array manipulation, but is quite slow. Typically, these
functions will be symbolically evaluated and compiled, though large arrays can take a very long time to compile due to
treating them as arrays of symbols, and not symbolic arrays. In practice, as long as compilation succeeds in a
reasonable time frame, performance of compiled code remains good.

This backend also addresses standing issues with type transparency in `juliacall` and `diffeqpy`'s use of it for
jit compilation. See here for more information: https://github.com/SciML/juliatorch/issues/14

Note also that not all numpy operations are currently supported. See `tests/lang_interop/julianumpy/python_fixtures.py`
to see which operations are currently tested and supported.
