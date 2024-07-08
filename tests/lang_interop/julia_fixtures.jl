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

function create_symbol(v:: Union{Array, PyArray}, name:: String)
    sym = Symbol(name)
    dims = size(v)
    slices = [1:d for d in dims]
    # Returns a symbolic array.
    return @variables $(sym)[slices...]
#     # Returns an array of symbolics.
#     return Symbolics.variables(name, slices...)
end

function create_symbol(v:: Number, name:: String)
    return Symbolics.variables(name)[1]
end

function symbolically_compile_function(f, args...; out_shape::Tuple=(), kwargs...)
    args_sym = [create_symbol(a, "arg_$i") for (i, a) in enumerate(args)]

    kwargs_vals_sym = [create_symbol(v, "kw_$k") for (k, v) in kwargs]

    # Construct new dict with the same keys but now with symbolic values.
    kwargs_keys = collect(keys(kwargs))  # also used later to ensure kwargs passed later are ordered
    kwargs_sym = Dict([k=>v for (k, v) in zip(kwargs_keys, kwargs_vals_sym)])

    println("--------------------")
    println("args_sym: ", args_sym)
    println("args_sym type: ", typeof(args_sym))
    println("kwargs_vals_sym: ", kwargs_vals_sym)
    println("kwargs_vals_sym type: ", typeof(kwargs_vals_sym))
    println("kwargs_keys: ", kwargs_keys)
    println("kwargs_sym: ", kwargs_sym)

    # Instantiate symbolic output array if we're expecting to return an array.
    f_computes_array = length(out_shape) > 0
    if f_computes_array
        # TODO support other types.
        # Note that this won't be an array of float64s, but rather an array of float64 symbols.
        out_sym = create_symbol(zeros(Float64, out_shape), "out")
    else
        out_sym = nothing
    end

    # Construct the symbolic representation of the function
    expr = f(args_sym...; out=out_sym, kwargs_sym...)

    if f_computes_array
        expr = out_sym
    end

    println("--------------------")
    println("preconv expr type: ", typeof(expr))
    println("preconv expr: ", expr)

    # If f is a python function, we'll need to unwrap the returned expression.
    if ispy(expr)
        expr = pyconvert(Any, expr)
    end

    println("--------------------")
    println("postconv expr type: ", typeof(expr))
    println("postconv expr: ", expr)
    println("postconv expr shape: ", size(expr))

    if ispy(expr)
        error("Could not convert python expression to julia expression.")
    end

    # Generate the compiled function, passing kwargs_vals_sym as arguments.
    sym_function = Symbolics.build_function(expr, args_sym..., kwargs_vals_sym...)

    println("--------------------")
    println("preconv sym_function type: ", typeof(sym_function))
    println("sym_function: ", sym_function)

    if f_computes_array
        # In this case, the compiled function should be a tuple, and the first version will allocate
        #  an array to return while the second one will take an out. We're using the second in-place
        #  version as a way to pre-specify the return types on the julia side.
        sym_has_in_place_expr = typeof(sym_function) <: Tuple
        # If this isn't the case, error.
        if !sym_has_in_place_expr
            error("Expected sym_function to be a tuple, but got: ", sym_function)
        end
        sym_function = sym_has_in_place_expr ? sym_function[2] : sym_function
    end

    println("--------------------")
    println("sym_function type: ", typeof(sym_function))
    println("sym_function: ", sym_function)

    compiled_function = eval(sym_function)

    println("--------------------")
    println("typeof compiled_function: ", typeof(compiled_function))
    println("compiled_function: ", compiled_function)

    # Now, we need to return a function with signature matching that of f.
    function resigged_f(fargs...; fkwargs...)
        # Map the fkwargs, which may have been passed in a different order, back onto the
        #  original passed in when the function was compiled.
        reordered_fkwargs_vals = [fkwargs[k] for k in kwargs_keys]

        println("--------------------")
        println("fargs: ", fargs)
        println("fkwargs: ", fkwargs)
        println("reordered_fkwargs_vals: ", reordered_fkwargs_vals)

        # Now we can pass the fargs and fkwargs_vals to the compiled function as regular arguments.
        if f_computes_array
            # TODO support other types.
            out = Array{Float64}(undef, out_shape...)
            Base.invokelatest(compiled_function, out, fargs..., reordered_fkwargs_vals...)
            return out
        else
            return Base.invokelatest(compiled_function, fargs..., reordered_fkwargs_vals...)
        end
    end

    return resigged_f
end
