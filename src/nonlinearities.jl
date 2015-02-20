### Activation functions.

function activation_function_selector(activation::ASCIIString)
	if activation == "sigmoid"
		return "sigmoid", sigmoid_activation!, sigmoid_jacobian
	elseif activation == "softmax"
		return "softmax", softmax_activation!, softmax_jacobian
	# Else default to linear activations.
	else
		return "linear", linear_activation!, linear_jacobian
	end
	# To be implemented:
	# exponential, leaky_rectified_linear, rectified_linear, softplus, tanh
end

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

function softmax_activation!{T<:FloatingPoint}(NET::Matrix{T}, ACT::Matrix{T})
	@inbounds begin
		for j = 1:size(NET, 2)
			# Get maximum net value (for numerical stability).
			maxnet = -Inf
			for i = 1:size(NET, 1)
				if NET[i, j] > maxnet
					maxnet = NET[i, j]
				end
			end
			# Get sum of exponentials.
			expsum = 0.0
			for i = 1:size(NET, 1)
				ACT[i, j] = exp(NET[i, j] - maxnet)
				expsum += ACT[i, j]
			end
			# Set activations.
			for i = 1:size(NET, 1)
				ACT[i, j] /= expsum
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


# function activation_exponential!{T<:FloatingPoint}(NET::Matrix{T}, ACT::Matrix{T}, DACT_DNET::Matrix{T})
# 	@inbounds begin
# 		for j = 1:size(NET, 2)
# 			for i = 1:size(NET, 1)
# 				expnet = exp(NET[i, j])
# 				ACT[i, j] = expnet
# 				DACT_DNET[i, j] = expnet
# 			end
# 		end
# 	end
# end
#
# function activation_linear!{T<:FloatingPoint}(NET::Matrix{T}, ACT::Matrix{T}, DACT_DNET::Matrix{T})
# 	@inbounds begin
# 		for j = 1:size(NET, 2)
# 			for i = 1:size(NET, 1)
# 				ACT[i, j] = NET[i, j]
# 				DACT_DNET[i, j] = 1.0
# 			end
# 		end
# 	end
# end
#
# function activation_rectified_linear!{T<:FloatingPoint}(NET::Matrix{T}, ACT::Matrix{T}, DACT_DNET::Matrix{T})
# 	@inbounds begin
# 		for j = 1:size(NET, 2)
# 			for i = 1:size(NET, 1)
# 				if NET[i, j] > 0.0
# 					ACT[i, j] = NET[i, j]
# 					DACT_DNET[i, j] = 1.0
# 				else
# 					ACT[i, j] = 0.0
# 					DACT_DNET[i, j] = 0.0
# 				end
# 			end
# 		end
# 	end
# end
#
# function activation_softplus!{T<:FloatingPoint}(NET::Matrix{T}, ACT::Matrix{T}, DACT_DNET::Matrix{T})
# 	@inbounds begin
# 		expx::T = 0.0
# 		for j = 1:size(NET, 2)
# 			for i = 1:size(NET, 1)
# 				expx = exp(NET[i, j])
# 				ACT[i, j] = log(1.0 + expx)
# 				DACT_DNET[i, j] = expx / (1.0 + expx)
# 			end
# 		end
# 	end
# end
#
# function activation_tanh!{T<:FloatingPoint}(NET::Matrix{T}, ACT::Matrix{T}, DACT_DNET::Matrix{T})
# 	@inbounds begin
# 		for j = 1:size(NET, 2)
# 			for i = 1:size(NET, 1)
# 				ACT[i, j] = tanh(NET[i, j])
# 				DACT_DNET[i, j] = 1.0 - (ACT[i, j] * ACT[i, j])
# 			end
# 		end
# 	end
# end

### Error functions.

function error_function_selector(error::ASCIIString)
	if error == "cross_entropy"
		return ("cross_entropy", cross_entropy!)
	else
		return ("squared_error", squared_error!)
	end
end

function cross_entropy!{T<:FloatingPoint}(YH::Matrix{T}, Y::Matrix{T}, E::Matrix{T}, DE_DYH::Matrix{T})
	@inbounds begin
		for j = 1:size(Y, 2)
			for i = 1:size(Y, 1)
				E[i, j] = -((Y[i, j] * log(YH[i, j])) + ((1.0 - Y[i, j]) * log(1.0 - YH[i, j])))
				DE_DYH[i, j] = ((1.0 - Y[i, j]) / (1.0 - YH[i, j])) - (Y[i, j] / YH[i, j])
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