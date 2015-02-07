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
		# Get sum.
		expsum::T = 0.0
		for i = 1:length(NET)
			expsum += exp(NET[i])
		end
		# Set normalized activations and gradient information.
		for i = 1:length(NET)
			expnet = exp(NET[i])
			ACT[i] = expnet / expsum
			DACT_DNET[i] = (expnet * (expsum - expnet)) / (expsum * expsum)
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

function error_function_selector(error_type::ASCIIString)
	if error_type == "cross_entropy"
		return ("cross_entropy", cross_entropy!)
	else
		return ("squared_error", squared_error!)
	end
end

function cross_entropy!{T<:FloatingPoint}(YH::Vector{T}, Y::Vector{T}, E::Vector{T}, DE_DYH::Vector{T})
	@inbounds begin
		for i = 1:length(YH)
			y, yh = Y[i], YH[i]
			E[i] = -((y * log(yh)) + ((1.0-y) * log(1.0-yh)))
			DE_DYH[i] = ((1.0-y) / (1.0-yh)) - (y/yh)
		end
	end
end

function squared_error!{T<:FloatingPoint}(YH::Vector{T}, Y::Vector{T}, E::Vector{T}, DE_DYH::Vector{T})
	@inbounds begin
		for i = 1:length(YH)
			E[i] = 0.5 * abs2(YH[i] - Y[i])
			DE_DYH[i] = YH[i] - Y[i]
		end
	end
end