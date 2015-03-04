### Error functions.

function error_function_selector(error::ASCIIString)
	if error == "cross_entropy"
		return "cross_entropy", cross_entropy!, cross_entropy_prime!
	else
		return "squared_error", squared_error!, squared_error_prime!
	end
end

function cross_entropy!{T<:FloatingPoint}(YH::AbstractVector{T}, Y::AbstractVector{T}, E::AbstractVector{T}; eta::T=1e-10)
	@inbounds begin
		for i = 1:length(Y)
			E[i] = -((Y[i] * log(YH[i] + eta)) + ((1.0 - Y[i]) * log(1.0 - YH[i] + eta)))
		end
	end
end

function cross_entropy_prime!{T<:FloatingPoint}(YH::AbstractVector{T}, Y::AbstractVector{T}, DE_DYH::AbstractVector{T}; eta::T=1e-10)
	@inbounds begin
		for i = 1:length(Y)
			DE_DYH[i] = ((1.0 - Y[i]) / (1.0 - YH[i] + eta)) - (Y[i] / (YH[i] + eta))
		end
	end
end

function squared_error!{T<:FloatingPoint}(YH::AbstractVector{T}, Y::AbstractVector{T}, E::AbstractVector{T})
	@inbounds begin
		for i = 1:length(Y)
			E[i] = 0.5 * abs2(YH[i] - Y[i])
		end
	end
end

function squared_error_prime!{T<:FloatingPoint}(YH::AbstractVector{T}, Y::AbstractVector{T}, DE_DYH::AbstractVector{T})
	@inbounds begin
		for i = 1:length(Y)
			DE_DYH[i] = YH[i] - Y[i]
		end
	end
end