### Activation functions.

function activation_function_selector(activation::ASCIIString)
	# To be implemented: exponential, leaky_rectified_linear, rectified_linear, softplus, tanh
	if activation == "exponential"
		return activation, exponential_activation!, exponential_jacobian, false
	elseif activation == "rectified_linear"
		return activation, rectified_linear_activation!, rectified_linear_jacobian, false
	elseif activation == "sigmoid"
		return activation, sigmoid_activation!, sigmoid_jacobian, false
	elseif activation == "softmax"
		return activation, softmax_activation!, softmax_jacobian, true
	elseif activation == "softplus"
		return activation, softplus_activation!, softplus_jacobian, false
	elseif activation == "tanh"
		return activation, tanh_activation!, tanh_jacobian, false
	# Else default to linear activations.
	else
		return "linear", linear_activation!, linear_jacobian, false
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

function exponential_jacobian{T<:FloatingPoint}(o::Int, oo::Int, NET::Vector{T}, ACT::Vector{T})
	o == oo ? ACT[o] : 0.0
end

### Linear activations.

function linear_activation!{T<:FloatingPoint}(NET::Vector{T}, ACT::Vector{T})
	@inbounds begin
		for i = 1:length(NET)
			ACT[i] = NET[i]
		end
	end
end

function linear_jacobian{T<:FloatingPoint}(o::Int, oo::Int, NET::Vector{T}, ACT::Vector{T})
	o == oo ? 1.0 : 0.0
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

function rectified_linear_jacobian{T<:FloatingPoint}(o::Int, oo::Int, NET::Vector{T}, ACT::Vector{T})
	(o == oo) && (NET[o] > 0.0) ? 1.0 : 0.0
end

### Sigmoid activations.

function sigmoid_activation!{T<:FloatingPoint}(NET::Vector{T}, ACT::Vector{T})
	@inbounds begin
		for i = 1:length(NET)
			ACT[i] = 1.0 / (1.0 + exp(-NET[i]))
		end
	end
end

function sigmoid_jacobian{T<:FloatingPoint}(o::Int, oo::Int, NET::Vector{T}, ACT::Vector{T})
	o == oo ? ACT[o] * (1.0 - ACT[o]) : 0.0
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

function softmax_jacobian{T<:FloatingPoint}(o::Int, oo::Int, NET::Vector{T}, ACT::Vector{T})
	if o == oo
		return ACT[o] * (1.0 - ACT[o])
	else
		return -ACT[o] * ACT[oo]
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

function softplus_jacobian{T<:FloatingPoint}(o::Int, oo::Int, NET::Vector{T}, ACT::Vector{T})
	if o == oo
		expx = exp(NET[o])
		return expx / (1.0 + expx)
	else
		return 0.0
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

function tanh_jacobian{T<:FloatingPoint}(o::Int, oo::Int, NET::Vector{T}, ACT::Vector{T})
	o == oo ? 1.0 - ACT[o] * ACT[o] : 0.0
end