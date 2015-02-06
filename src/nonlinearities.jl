### Activation functions.

function activation_function_selector(activation_type::ASCIIString)
	if activation_type == "exponential"
		return ("exponential", activation_exponential!)
	elseif activation_type == "rectified_linear"
		return ("rectified_linear", activation_rectified_linear!)
	elseif activation_type == "sigmoid"
		return ("sigmoid", activation_sigmoid!)
	elseif activation_type == "softmax"
		return ("softmax", activation_softmax!)
	elseif activation_type == "tanh"
		return ("tanh", activation_tanh!)
	else
		return ("linear", activation_linear!)
	end
end

function activation_exponential!{T<:FloatingPoint}(NET::Vector{T}, ACT::Vector{T}, DACT_DNET::Vector{T})
	@inbounds begin
		for i = 1:length(NET)
			expnet = exp(NET[i])
			ACT[i] = expnet
			DACT_DNET[i] = expnet
		end
	end
end

function activation_linear!{T<:FloatingPoint}(NET::Vector{T}, ACT::Vector{T}, DACT_DNET::Vector{T})
	@inbounds begin
		for i = 1:length(NET)
			ACT[i] = NET[i]
			DACT_DNET[i] = 1.0
		end
	end
end

function activation_rectified_linear!{T<:FloatingPoint}(NET::Vector{T}, ACT::Vector{T}, DACT_DNET::Vector{T})
	@inbounds begin
		for i = 1:length(NET)
			if NET[i] > 0.0
				ACT[i] = NET[i]
				DACT_DNET[i] = 1.0
			else
				ACT[i] = 0.0
				DACT_DNET[i] = 0.0
			end
		end
	end
end

function activation_sigmoid!{T<:FloatingPoint}(NET::Vector{T}, ACT::Vector{T}, DACT_DNET::Vector{T})
	@inbounds begin
		for i = 1:length(NET)
			expnx = exp(-NET[i])
			ACT[i] = 1.0 / (1.0 + expnx)
			tmp = 1.0 + expnx
			DACT_DNET[i] = expnx / (tmp * tmp)
		end
	end
end

function activation_softmax!{T<:FloatingPoint}(NET::Vector{T}, ACT::Vector{T}, DACT_DNET::Vector{T})
	@inbounds begin
		# Exponentiate and compute sum.
		sum = 0.0
		for i = 1:length(NET)
			ACT[i] = exp(NET[i])
			sum += ACT[i]
		end
		# Normalize.
		if sum > 0.0
			for i = 1:length(ACT)
				ACT[i] /= sum
			end
		end
		# Set gradient.
		for i = 1:length(NET)
			expx = exp(NET[i])
			DACT_DNET[i] = (expx*sum - expx*expx) / (sum * sum)
		end
	end
end

function activation_tanh!{T<:FloatingPoint}(NET::Vector{T}, ACT::Vector{T}, DACT_DNET::Vector{T})
	@inbounds begin
		for i = 1:length(NET)
			ACT[i] = tanh(NET[i])
			DACT_DNET[i] = 1.0 - (ACT[i] * ACT[i])
		end
	end
end

### Error functions.

function error_squared!{T<:FloatingPoint}(YH::Vector{T}, Y::Vector{T}, OUT::Vector{T})
	@inbounds begin
		for i = 1:length(Y)
			OUT[i] = 0.5 * ((YH[i] - Y[i]) .^ 2)
		end
	end
end