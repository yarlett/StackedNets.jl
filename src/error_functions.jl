### Error functions.

function error_function_selector(error::ASCIIString)
	if error == "cross_entropy"
		return ("cross_entropy", cross_entropy!)
	else
		return ("squared_error", squared_error!)
	end
end

function cross_entropy!{T<:FloatingPoint}(YH::Matrix{T}, Y::Matrix{T}, E::Matrix{T}, DE_DYH::Matrix{T}; eta::T=1e-10)
	@inbounds begin
		for j = 1:size(Y, 2)
			for i = 1:size(Y, 1)
				E[i, j] = -((Y[i, j] * log(YH[i, j] + eta)) + ((1.0 - Y[i, j]) * log(1.0 - YH[i, j] + eta)))
				DE_DYH[i, j] = ((1.0 - Y[i, j]) / (1.0 - YH[i, j] + eta)) - (Y[i, j] / (YH[i, j] + eta))
			end
		end
	end
end

function squared_error!{T<:FloatingPoint}(YH::Matrix{T}, Y::Matrix{T}, E::Matrix{T}, DE_DYH::Matrix{T})
	@inbounds begin
		for j = 1:size(Y, 2)
			for i = 1:size(Y, 1)
				E[i, j] = 0.5 * abs2(YH[i, j] - Y[i, j])
				DE_DYH[i, j] = YH[i, j] - Y[i, j]
			end
		end
	end
end