# Here we need julia functions, each of which takes a function and its args, kwargs, and then
# 1. forward evaluates it, returning the result.
# 2. evaluates it with forward diff, returning the result.
# 3. evaluates it with backward diff, returning the result.
# 4. evaluates it after converting the inputs to Symbolics.jl variables, and then returning
#     the results.

# First, get the relevant packages.
using ForwardDiff
using Symbolics
using SymbolicUtils
using Base
using PythonCall

forward_eval(f, args...; kwargs...) = f(args...; kwargs...)
forward_diff_eval(f, args...; kwargs...) = ForwardDiff.gradient(f, args...; kwargs...)

function symbolically_compile_function(f, args...; kwargs...)
    @variables args_arr_sym[1:length(args)]
    args_sym = Symbolics.scalarize(args_arr_sym)

    @variables kwargs_vals_arr_sym[1:length(kwargs)]
    kwargs_vals_sym = Symbolics.scalarize(kwargs_vals_arr_sym)

    # Construct new dict with the same keys but now with symbolic values.
    kwargs_keys = collect(keys(kwargs))  # also used later to ensure kwargs passed later are ordered
    kwargs_sym = Dict([k=>v for (k, v) in zip(kwargs_keys, kwargs_vals_sym)])

#     println("--------------------")
#     println("args_sym: ", args_sym)
#     println("args_sym type: ", typeof(args_sym))
#     println("kwargs_vals_sym: ", kwargs_vals_sym)
#     println("kwargs_vals_sym type: ", typeof(kwargs_vals_sym))
#     println("kwargs_keys: ", kwargs_keys)
#     println("kwargs_sym: ", kwargs_sym)
#     println("--------------------")

    # Construct the symbolic representation of the function
    expr = f(args_sym...; kwargs_sym...)
    # If f is a python function, we'll need to unwrap the returned expression.
    if typeof(expr) == Py
        expr = pyconvert(Num, expr)
    end

#     println("--------------------")
#     println("expr type: ", typeof(expr))
#     println("expr: ", expr)
#     println("--------------------")

    # Generate the compiled function, passing kwargs_vals_sym as arguments.
    sym_function = Symbolics.build_function(expr, args_arr_sym, kwargs_vals_sym)

#     println("--------------------")
#     println("sym_function: ", sym_function)
#     println("--------------------")

    compiled_function = eval(sym_function)

#     println("--------------------")
#     println("compiled_function: ", compiled_function)
#     println("--------------------")

    # Now, we need to return a function with signature matching that of f.
    function resigged_f(fargs...; fkwargs...)
        # Map the fkwargs, which may have been passed in a different order, back onto the
        #  original passed in when the function was compiled.
        reordered_fkwargs_vals = [fkwargs[k] for k in kwargs_keys]

#         println("--------------------")
#         println("fargs: ", fargs)
#         println("fkwargs: ", fkwargs)
#         println("reordered_fkwargs_vals: ", reordered_fkwargs_vals)
#         println("--------------------")

        # Now we can pass the fargs and fkwargs_vals to the compiled function as regular arguments.
        return Base.invokelatest(compiled_function, fargs, reordered_fkwargs_vals)
    end

    return resigged_f
end

function symbolic_forward_eval(f, args...; kwargs...)
    compiled_f = symbolically_compile_function(f, args...; kwargs...)
    return compiled_f(args...; kwargs...)
end
