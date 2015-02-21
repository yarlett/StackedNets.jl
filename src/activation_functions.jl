### Activation functions.

function activation_function_selector(activation::ASCIIString)
	# To be implemented: exponential, leaky_rectified_linear, rectified_linear, softplus, tanh
	if activation == "exponential"
		return activation, exponential_activation!, exponential_jacobian
	elseif activation == "rectified_linear"
		return activation, rectified_linear_activation!, rectified_linear_jacobian
	elseif activation == "sigmoid"
		return activation, sigmoid_activation!, sigmoid_jacobian
	elseif activation == "softmax"
		return activation, softmax_activation!, softmax_jacobian
	elseif activation == "softplus"
		return activation, softplus_activation!, softplus_jacobian
	elseif activation == "tanh"
		return activation, tanh_activation!, tanh_jacobian
	# Else default to linear activations.
	else
		return "linear", linear_activation!, linear_jacobian
	end
end

### Exponential activations.

function exponential_activation!{T<:FloatingPoint}(NET::Matrix{T}, ACT::Matrix{T})
	@inbounds begin
		for j = 1:size(NET, 2)
			for i = 1:size(NET, 1)
				ACT[i, j] = exp(NET[i, j])
			end
		end	
	end
end

function exponential_jacobian{T<:FloatingPoint}(o::Int, oo::Int, NET::Vector{T}, ACT::Vector{T})
	o == oo ? ACT[o] : 0.0
end

### Linear activations.

function linear_activation!{T<:FloatingPoint}(NET::Matrix{T}, ACT::Matrix{T})
	@inbounds begin
		for j = 1:size(NET, 2)
			for i = 1:size(NET, 1)
				ACT[i, j] = NET[i, j]
			end
		end	
	end
end

function linear_jacobian{T<:FloatingPoint}(o::Int, oo::Int, NET::Vector{T}, ACT::Vector{T})
	o == oo ? 1.0 : 0.0
end

### Rectified linear activations.

function rectified_linear_activation!{T<:FloatingPoint}(NET::Matrix{T}, ACT::Matrix{T})
	@inbounds begin
		for j = 1:size(NET, 2)
			for i = 1:size(NET, 1)
				if NET[i, j] > 0.0
					ACT[i, j] = NET[i, j]
				else
					ACT[i, j] = 0.0
				end
			end
		end	
	end
end

function rectified_linear_jacobian{T<:FloatingPoint}(o::Int, oo::Int, NET::Vector{T}, ACT::Vector{T})
	(o == oo) && (NET[o] > 0.0) ? 1.0 : 0.0
end

### Sigmoid activations.

function sigmoid_activation!{T<:FloatingPoint}(NET::Matrix{T}, ACT::Matrix{T})
	@inbounds begin
		for j = 1:size(NET, 2)
			for i = 1:size(NET, 1)
				ACT[i, j] = 1.0 / (1.0 + exp(-NET[i, j]))
			end
		end	
	end
end

function sigmoid_jacobian{T<:FloatingPoint}(o::Int, oo::Int, NET::Vector{T}, ACT::Vector{T})
	o == oo ? ACT[o] * (1.0 - ACT[o]) : 0.0
end

### Softmax activations.

function softmax_activation!{T<:FloatingPoint}(NET::Matrix{T}, ACT::Matrix{T})
	@inbounds begin
		for j = 1:size(NET, 2)
			# Get maximum net value (for numerical stability).
			maxnet::T = -Inf
			for i = 1:size(NET, 1)
				if NET[i, j] > maxnet
					maxnet = NET[i, j]
				end
			end
			# Get sum of exponentials.
			expsum::T = 0.0
			for i = 1:size(NET, 1)
				ACT[i, j] = exp(NET[i, j] - maxnet)
				expsum += ACT[i, j]
			end
			# Set activations.
			for i = 1:size(NET, 1)
				ACT[i, j] ./= expsum
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

function softplus_activation!{T<:FloatingPoint}(NET::Matrix{T}, ACT::Matrix{T})
	@inbounds begin
		for j = 1:size(NET, 2)
			for i = 1:size(NET, 1)
				ACT[i, j] = log(1.0 + exp(NET[i, j]))
			end
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

function tanh_activation!{T<:FloatingPoint}(NET::Matrix{T}, ACT::Matrix{T})
	@inbounds begin
		for j = 1:size(NET, 2)
			for i = 1:size(NET, 1)
				ACT[i, j] = tanh(NET[i, j])
			end
		end	
	end
end

function tanh_jacobian{T<:FloatingPoint}(o::Int, oo::Int, NET::Vector{T}, ACT::Vector{T})
	o == oo ? 1.0 - ACT[o] * ACT[o] : 0.0
end