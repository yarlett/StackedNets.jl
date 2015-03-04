### Activation functions.

function activation_function_selector(activation::ASCIIString)
	if activation == "exponential"
		return activation, exponential_activation!, exponential_backward!
	elseif activation == "leaky_rectified_linear"
		return activation, leaky_rectified_linear_activation!, leaky_rectified_linear_backward!
	elseif activation == "rectified_linear"
		return activation, rectified_linear_activation!, rectified_linear_backward!
	elseif activation == "sigmoid"
		return activation, sigmoid_activation!, sigmoid_backward!
	elseif activation == "softmax"
		return activation, softmax_activation!, softmax_backward!
	elseif activation == "softplus"
		return activation, softplus_activation!, softplus_backward!
	elseif activation == "tanh"
		return activation, tanh_activation!, tanh_backward!
	# Else default to linear activations.
	else
		return "linear", linear_activation!, linear_backward!
	end
end

### Exponential activations.

function exponential_activation!{T<:FloatingPoint}(NET::Vector{T}, ACT::Vector{T})
	@inbounds begin
		for i = 1:length(NET)
			ACT[i] = exp(NET[i])
		end
	end
end

function exponential_backward!{T<:FloatingPoint}(NET::Vector{T}, ACT::Vector{T}, DELTAS_ABOVE::Vector{T}, DE_DNET::Vector{T})
	@inbounds begin
		for o = 1:length(NET)
			DE_DNET[o] = DELTAS_ABOVE[o] * ACT[o]
		end
	end
end

### Linear activations.

function linear_activation!{T<:FloatingPoint}(NET::Vector{T}, ACT::Vector{T})
	@inbounds begin
		for i = 1:length(NET)
			ACT[i] = NET[i]
		end
	end
end

function linear_backward!{T<:FloatingPoint}(NET::Vector{T}, ACT::Vector{T}, DELTAS_ABOVE::Vector{T}, DE_DNET::Vector{T})
	@inbounds begin
		for o = 1:length(NET)
			DE_DNET[o] = DELTAS_ABOVE[o]
		end
	end
end

### Leaky rectified linear activations.

function leaky_rectified_linear_activation!{T<:FloatingPoint}(NET::Vector{T}, ACT::Vector{T})
	@inbounds begin
		for i = 1:length(NET)
			if NET[i] > 0.0
				ACT[i] = NET[i]
			else
				ACT[i] = 0.01 * NET[i]
			end
		end
	end
end

function leaky_rectified_linear_backward!{T<:FloatingPoint}(NET::Vector{T}, ACT::Vector{T}, DELTAS_ABOVE::Vector{T}, DE_DNET::Vector{T})
	@inbounds begin
		for o = 1:length(NET)
			if NET[o] > 0.0
				DE_DNET[o] = DELTAS_ABOVE[o]
			else
				DE_DNET[o] = DELTAS_ABOVE[o] * 0.01
			end
		end
	end
end

### Rectified linear activations.

function rectified_linear_activation!{T<:FloatingPoint}(NET::Vector{T}, ACT::Vector{T})
	@inbounds begin
		for i = 1:length(NET)
			if NET[i] > 0.0
				ACT[i] = NET[i]
			else
				ACT[i] = 0.0
			end
		end
	end
end

function rectified_linear_backward!{T<:FloatingPoint}(NET::Vector{T}, ACT::Vector{T}, DELTAS_ABOVE::Vector{T}, DE_DNET::Vector{T})
	@inbounds begin
		for o = 1:length(NET)
			if NET[o] > 0.0
				DE_DNET[o] = DELTAS_ABOVE[o]
			else
				DE_DNET[o] = 0.0
			end
		end
	end
end

### Sigmoid activations.

function sigmoid_activation!{T<:FloatingPoint}(NET::Vector{T}, ACT::Vector{T})
	@inbounds begin
		for i = 1:length(NET)
			ACT[i] = 1.0 / (1.0 + exp(-NET[i]))
		end
	end
end

function sigmoid_backward!{T<:FloatingPoint}(NET::Vector{T}, ACT::Vector{T}, DELTAS_ABOVE::Vector{T}, DE_DNET::Vector{T})
	@inbounds begin
		for o = 1:length(NET)
			DE_DNET[o] = DELTAS_ABOVE[o] * ACT[o] * (1.0 - ACT[o])
		end
	end
end

### Softmax activations.

function softmax_activation!{T<:FloatingPoint}(NET::Vector{T}, ACT::Vector{T})
	@inbounds begin
		for j = 1:size(NET, 2)
			# Get maximum net value (for numerical stability).
			maxnet::T = -Inf
			for i = 1:length(NET)
				if NET[i] > maxnet
					maxnet = NET[i]
				end
			end
			# Get sum of exponentials.
			expsum::T = 0.0
			for i = 1:length(NET)
				ACT[i] = exp(NET[i] - maxnet)
				expsum += ACT[i]
			end
			# Set activations.
			for i = 1:length(NET)
				ACT[i] ./= expsum
			end
		end
	end
end

function softmax_backward!{T<:FloatingPoint}(NET::Vector{T}, ACT::Vector{T}, DELTAS_ABOVE::Vector{T}, DE_DNET::Vector{T})
	@inbounds begin
		no = length(NET)
		for o = 1:no
			DE_DNET[o] = 0.0
			for oo = 1:no
				if o == oo
					DE_DNET[o] += DELTAS_ABOVE[oo] * ACT[o] * (1.0 - ACT[o])
				else
					DE_DNET[o] -= DELTAS_ABOVE[oo] * ACT[o] * ACT[oo]
				end
			end
		end
	end
end

### Softplus activations.

function softplus_activation!{T<:FloatingPoint}(NET::Vector{T}, ACT::Vector{T})
	@inbounds begin
		for i = 1:length(NET)
			ACT[i] = log(1.0 + exp(NET[i]))
		end
	end
end

function softplus_backward!{T<:FloatingPoint}(NET::Vector{T}, ACT::Vector{T}, DELTAS_ABOVE::Vector{T}, DE_DNET::Vector{T})
	@inbounds begin
		for o = 1:length(NET)
			expx = exp(NET[o])
			DE_DNET[o] = DELTAS_ABOVE[o] * expx / (1.0 + expx)
		end
	end
end

### Tanh activations.

function tanh_activation!{T<:FloatingPoint}(NET::Vector{T}, ACT::Vector{T})
	@inbounds begin
		for i = 1:length(NET)
			ACT[i] = tanh(NET[i])
		end
	end
end

function tanh_backward!{T<:FloatingPoint}(NET::Vector{T}, ACT::Vector{T}, DELTAS_ABOVE::Vector{T}, DE_DNET::Vector{T})
	@inbounds begin
		for o = 1:length(NET)
			DE_DNET[o] = DELTAS_ABOVE[o] * (1.0 - ACT[o] * ACT[o])
		end
	end
end
