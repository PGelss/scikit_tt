using PyCall
using TensorOperations
using Base.Threads

#tensor_train = pyimport("scikit_tt.tensor_train")

#TT = pytype_query(tensor_train.TT)

export core_mult
export core_mult_col
export tensor_train_multiplication


function core_mult_col(core_1, core_2)

	r1, m, n, r2 = size(core_1)
	s1, n, p, s2 = size(core_2)
    
	dim_1 = r1 * s1
	dim_2 = m
	dim_3 = p
	dim_4 = r2 * s2
    
	# Initialize new shape
	result_shape = (dim_1, dim_2, dim_3, dim_4)

	# Pre-allocate memory for resulting core
	#result = zeros(Float64, result_shape)
    result = Array{Float64}(undef, (r1, s1, m, p, r2, s2))
    #result = zeros(Float64, result_shape)

    for j2 = 1:s2, i2 = 1:r2, j1 = 1:s1, i1 = 1:r1

        i = (i1-1) * s1 + j1
        j = (i2-1) * s2 + j2

        result[i, :, :, j] = core_1[i1, :, :, i2] * core_2[j1, :, :, j2]

    end

    #perm_core_1 = permutedims(core_1, (1, 4, 2, 3))
    #perm_core_2 = permutedims(core_2, (1, 4, 2, 3))

    #for j = 1:r1, l = 1:s1, i = 1:r2, k = 1:s2

    #    result = perm_core_1[:, :, i, j] * perm_core_2[:, :, k, l]

    #end

        
        

    return result
end 


function core_mult(core_1::Array{Float64, 4}, core_2::Array{Float64, 4})::Array{Float64}

	r1, m, n, r2 = size(core_1)
	s1, n, p, s2 = size(core_2)
    
	dim_1 = r1 * s1
	dim_2 = m
	dim_3 = p
	dim_4 = r2 * s2
    
	# Initialize new shape
	result_shape = (r1, s1, m, p, r2, s2)

	# Pre-allocate memory for resulting core
	#result = zeros(Float64, result_shape)

    @tensor result[a, e, b, f, d, g] := core_1[a, b, c, d] * core_2[e, c, f, g]

    result = reshape(result, (dim_1, dim_2, dim_3, dim_4))

    return result
end

function tensor_train_multiplication(lhs, rhs)
    
    cores = [ core_mult(lhs[i], rhs[i]) for i = 1:length(lhs) ]

    return cores
end

    
