TODO describe the backend abstraction and how someone can implement a new/alternative one.

1. put meat in an internals file or folder.
2. define conversions in julia_to_python_rules and python_to_julia_rules using single dispatch.
3. can override default implementation by registering an overload with type object.

Note that even though we call it julia_to_python_rules, these are python-side conversions from not-very-robust
julia-call-wrapped objects to more robust wrapped objects. python_to_julia_rules are similar in that they return
things to juliacall types. So really, these are juliacall_to_python and python_to_juliacall rules.
And...even then, we really are interested in `juliacall_juliasymbolics_to/from_python`...


TODO probably rename this to juliacallnumpy, and rename the respective conversion rules.