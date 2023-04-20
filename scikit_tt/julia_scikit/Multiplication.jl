module Multiplication

using TensorOperations

export core_mult

function core_mult(core_1, core_2)

	r1, m, n, r2 = size(core_1)
	s1, n, p, s2 = size(core_2)
    
	dim_1 = r1 * s1
	dim_2 = m
	dim_3 = p
	dim_4 = r2 * s2
    
	# Initialize new shape
	result_shape = (r1, s1, m, p, r2, s2)

	# Pre-allocate memory for resulting core
	result = zeros(Float64, result_shape)

    @tensor result[a, e, b, f, d, g] = core_1[a, b, c, d] * core_2[e, c, f, g]

    result = reshape(result, (dim_1, dim_2, dim_3, dim_4))

    return result
end

end
