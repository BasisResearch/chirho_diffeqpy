# We need a function that takes Vector of len K of arrays, where the inner arrays have shape
#  (T, *spatial), where e.g. spatial might == (X, Y, Z). We need to flatten everything accordingly
#  into an array of shape (T, K * X * Y * Z).


function flatten_multi_domain_interpolation(interpolations::Vector{Array{Float64, N}}) where {N}

    # Start by just putting everything into a big array of shape (T, K, X, Y, Z) so that
    #  we can flatten it more easiliy with reshaping operations.
    # We assume each element of interpolations is of shape (T, X, Y, Z).
    # Note that X, Y, Z is just an example, it could be (X,) or (X, Z) or (X, Y, Z, L, M, N) etc.

    T = size(interpolations[1], 1)
    K = length(interpolations)
    spatial_shape = size(interpolations[1])[2:end]

    # Initialize the big array
    big_array = zeros(T, K, spatial_shape...)

    # Fill the big array
    colons = ntuple(_ -> Colon(), length(spatial_shape))
    for k in 1:K
        big_array[:, k, colons...] = interpolations[k]
    end

    # TODO WIP when this is reshaped back on the python side, I think it will actually want to
    #  unpack in to a shape of (T, Z, Y, X, K) due to column major ordering in julia.
    # This is confusing b/c the outer code (in internals.py) already transposes the julia
    #  (state, time) array into a (time, state) so that it can flatten on julia side
    #  and then unflatten into (state, time) on python side.
    # So here, we need to permute all but time, so that the outer code can handle the
    #  (state, time) unflatten, and this code can handle the unflattenability of the other dims.
    println("Andy fix the column-major row-major reshaping issues here.")

    # Now flatten the big array and return, and transpose, because the standard ODE solves
    #  return (state, time) shapes, despite time-first multi-domain settings returning (time, ...)
    #  shapes, and the outer code transposes into what python expects, which is (time, ...).
    return permutedims(reshape(big_array, (T, K * prod(spatial_shape))), (2, 1))

end
