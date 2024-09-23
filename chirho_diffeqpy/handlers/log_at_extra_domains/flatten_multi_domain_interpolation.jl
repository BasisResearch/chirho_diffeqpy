# We need a function that takes Vector of len K of arrays, where the inner arrays have shape
#  (T, *spatial), where e.g. spatial might == (X, Y, Z). We need to flatten everything accordingly
#  into an array of shape (T, K * X * Y * Z).


function flatten_multi_domain_interpolation(interpolations::Vector{Array{Float64, N}}) where {N}

    # Start by just putting everything into a big array of shape (T, K, X, Y, Z) so that
    #  we can flatten it more easiliy with reshaping operations.

    T = size(interpolations[1], 1)
    K = length(interpolations)
    spatial_shape = size(interpolations[1])[2:end]

    # Initialize the big array
    big_array = zeros(T, K, spatial_shape...)

    # Fill the big array
    for k in 1:K
        big_array[:, k, :] = interpolations[k]
    end

    # TODO WIP when this is reshaped back on the python side, I think it will actually want to
    #  unpack in to a shape of (T, Z, Y, X, K) due to column major ordering in julia.
    println("Andy fix the column-major row-major reshaping issues here.")

    # Now flatten the big array
    return reshape(big_array, (T, K * prod(spatial_shape)))

end
