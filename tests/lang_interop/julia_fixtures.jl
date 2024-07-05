# Here we need julia functions, each of which takes a function and its args, kwargs, and then
# 1. forward evaluates it, returning the result.
# 2. evaluates it with forward diff, returning the result.
# 3. evaluates it with backward diff, returning the result.
# 4. evaluates it after converting the inputs to Symbolics.jl variables, and then returning
#     the results.

# First, get the relevant packages.
using ForwardDiff
using Symbolics
using Base

forward_eval(f, args...; kwargs...) = f(args...; kwargs...)
forward_diff_eval(f, args...; kwargs...) = ForwardDiff.gradient(f, args...; kwargs...)



function symbolic_forward_eval(f, args...; kwargs...)

    # print f
    print("f: \n")
    print(f)
    print("\n------------------------------------------------------\n")

    # print args and kwargs
    print("args: \n")
    print(args)
    print("\n------------------------------------------------------\n")

    print("kwargs: \n")
    print(kwargs)
    print("\n------------------------------------------------------\n")

    # Create symbolic variables for args and kwargs
    sym_args = [@variables Symbol("arg_$i")[1] for i in 1:length(args)]

    print("sym_args: \n")
    print(sym_args...)
    print("\n------------------------------------------------------\n")

    sym_kwargs = Dict([Symbolics.@variables Symbol("kwarg_$(k)")[1] for k in keys(kwargs)])

    print("sym_kwargs: \n")
    print(sym_kwargs...)
    print("\n------------------------------------------------------\n")

    # Build the expression using the function f
    expr = f(sym_args..., ; (Symbol(k)=>v for (k, v) in sym_kwargs)...)

    print("expr: \n")
    print(expr)
    print("\n------------------------------------------------------\n")

    # Convert the expression to a function
    sym_function = Symbolics.build_function(expr, sym_args..., ; [v for v in values(sym_kwargs)]...)

    print("sym_function: \n")
    print(sym_function)
    print("\n------------------------------------------------------\n")

    # Compile the function
    compiled_function = eval(sym_function)

    print("compiled_function: \n")
    print(compiled_function)
    print("\n------------------------------------------------------\n")

    # print args and kwargs
    print("args: \n")
    print(args)
    print("\n------------------------------------------------------\n")

    print("kwargs: \n")
    print(kwargs)
    print("\n------------------------------------------------------\n")

    # Evaluate the function with original args and kwargs
    # result = compiled_function(args..., ; values(kwargs)...)
    result = Base.invokelatest(compiled_function, args...; values(kwargs)...)


    print("result: \n")
    print(result)
    print("\n------------------------------------------------------\n")

    return result
end
