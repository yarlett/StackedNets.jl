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
	elseif activation_type == "softplus"
		return ("softplus", activation_softplus!)
	elseif activation_type == "tanh"
		return ("tanh", activation_tanh!)
	else
		return ("linear", activation_linear!)
	end
end

function activation_exponential!{T<:FloatingPoint}(NET::Matrix{T}, ACT::Matrix{T}, DACT_DNET::Matrix{T})
	@inbounds begin
		for j = 1:size(NET, 2)
			for i = 1:size(NET, 1)
				expnet = exp(NET[i, j])
				ACT[i, j] = expnet
				DACT_DNET[i, j] = expnet
			end
		end
	end
end

function activation_linear!{T<:FloatingPoint}(NET::Matrix{T}, ACT::Matrix{T}, DACT_DNET::Matrix{T})
	@inbounds begin
		for j = 1:size(NET, 2)
			for i = 1:size(NET, 1)
				ACT[i, j] = NET[i, j]
				DACT_DNET[i, j] = 1.0
			end
		end
	end
end

function activation_rectified_linear!{T<:FloatingPoint}(NET::Matrix{T}, ACT::Matrix{T}, DACT_DNET::Matrix{T})
	@inbounds begin
		for j = 1:size(NET, 2)
			for i = 1:size(NET, 1)
				if NET[i, j] > 0.0
					ACT[i, j] = NET[i, j]
					DACT_DNET[i, j] = 1.0
				else
					ACT[i, j] = 0.0
					DACT_DNET[i, j] = 0.0
				end
			end
		end
	end
end

function activation_sigmoid!{T<:FloatingPoint}(NET::Matrix{T}, ACT::Matrix{T}, DACT_DNET::Matrix{T})
	@inbounds begin
		for j = 1:size(NET, 2)
			for i = 1:size(NET, 1)
				ACT[i, j] = 1.0 / (1.0 + exp(-NET[i, j]))
				DACT_DNET[i, j] = ACT[i, j] * (1.0 - ACT[i, j])
			end
		end
	end
end

function activation_softmax!{T<:FloatingPoint}(NET::Matrix{T}, ACT::Matrix{T}, DACT_DNET::Matrix{T})
	@inbounds begin
		I, J = size(NET)
		for j = 1:J
			# Get maximum net value.
			netmax::T = -Inf
			for i = 1:I
				if NET[i, j] > netmax
					netmax = NET[i, j]
				end
			end
			# Get sum of exponentials.
			expsum::T = 0.0
			for i = 1:I
				ACT[i, j] = exp(NET[i, j] - netmax)
				expsum += ACT[i, j]
			end
			# Compute normalized activations and gradient information.
			for i = 1:I
				ACT[i, j] = ACT[i, j] / expsum
				DACT_DNET[i, j] = ACT[i, j] * (1.0 - ACT[i, j])
			end
		end
	end
end

function activation_softplus!{T<:FloatingPoint}(NET::Matrix{T}, ACT::Matrix{T}, DACT_DNET::Matrix{T})
	@inbounds begin
		expx::T = 0.0
		for j = 1:size(NET, 2)
			for i = 1:size(NET, 1)
				expx = exp(NET[i, j])
				ACT[i, j] = log(1.0 + expx)
				DACT_DNET[i, j] = expx / (1.0 + expx)
			end
		end
	end
end

function activation_tanh!{T<:FloatingPoint}(NET::Matrix{T}, ACT::Matrix{T}, DACT_DNET::Matrix{T})
	@inbounds begin
		for j = 1:size(NET, 2)
			for i = 1:size(NET, 1)
				ACT[i, j] = tanh(NET[i, j])
				DACT_DNET[i, j] = 1.0 - (ACT[i, j] * ACT[i, j])
			end
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

function cross_entropy!{T<:FloatingPoint}(YH::Matrix{T}, Y::Matrix{T}, E::Matrix{T}, DE_DYH::Matrix{T})
	@inbounds begin
		for j = 1:size(YH, 2)
			for i = 1:size(YH, 1)
				y, yh = Y[i, j], YH[i, j]
				E[i, j] = -((y * log(yh)) + ((1.0-y) * log(1.0-yh)))
				DE_DYH[i, j] = ((1.0-y) / (1.0-yh)) - (y/yh)
			end
		end
	end
end

function squared_error!{T<:FloatingPoint}(YH::Matrix{T}, Y::Matrix{T}, E::Matrix{T}, DE_DYH::Matrix{T})
	@inbounds begin
		for j = 1:size(YH, 2)
			for i = 1:size(YH, 1)
				E[i, j] = 0.5 * abs2(YH[i, j] - Y[i, j])
				DE_DYH[i, j] = YH[i, j] - Y[i, j]
			end
		end
	end
end