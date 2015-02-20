### Error functions.

function error_function_selector(error::ASCIIString)
	if error == "cross_entropy"
		return "cross_entropy", cross_entropy!
	else
		return "squared_error", squared_error!
	end
end

function cross_entropy!{T<:FloatingPoint}(YH::Matrix{T}, Y::Matrix{T}, E::Matrix{T}, DE_DYH::Matrix{T}; eta::T=1e-10)
	@inbounds begin
		for i = 1:length(Y)
			E[i] = -((Y[i] * log(YH[i] + eta)) + ((1.0 - Y[i]) * log(1.0 - YH[i] + eta)))
			DE_DYH[i] = ((1.0 - Y[i]) / (1.0 - YH[i] + eta)) - (Y[i] / (YH[i] + eta))
		end
	end
end

function squared_error!{T<:FloatingPoint}(YH::Matrix{T}, Y::Matrix{T}, E::Matrix{T}, DE_DYH::Matrix{T})
	@inbounds begin
		for i = 1:length(Y)
			E[i] = 0.5 * abs2(YH[i] - Y[i])
			DE_DYH[i] = YH[i] - Y[i]
		end
	end
end