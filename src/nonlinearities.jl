### Activation functions.

function activation_function_selector(activation_type::AbstractString)
	if activation_type == "rectified_linear"
		activation_function! = activation_rectified_linear!
	elseif activation_type == "sigmoid"
		activation_function! = activation_sigmoid!
	elseif activation_type == "softmax"
		activation_function! = activation_softmax!
	elseif activation_type == "tanh"
		activation_function! = activation_tanh!
	else
		activation_function! = activation_linear!
	end
	activation_function!
end

function activation_linear!(NET::Vector{FloatingPoint}, ACT::Vector{FloatingPoint}, GRA::Vector{FloatingPoint})
	@inbounds begin
		for i = 1:length(NET)
			ACT[i] = NET[i]
			GRA[i] = 1.0
		end
	end
end

function activation_rectified_linear!(NET::Vector{FloatingPoint}, ACT::Vector{FloatingPoint}, GRA::Vector{FloatingPoint})
	@inbounds begin
		for i = 1:length(NET)
			if NET[i] > 0.0
				ACT[i] = NET[i]
				GRA[i] = 1.0
			else
				ACT[i] = 0.0
				GRA[i] = 0.0
			end
		end
	end
end

function activation_sigmoid!(NET::Vector{FloatingPoint}, ACT::Vector{FloatingPoint}, GRA::Vector{FloatingPoint})
	@inbounds begin
		for i = 1:length(NET)
			expnx = exp(-NET[i])
			ACT[i] = 1.0 / (1.0 + expnx)
			tmp = 1.0 + expnx
			GRA[i] = expnx / (tmp * tmp)
		end
	end
end

function activation_softmax!(NET::Vector{FloatingPoint}, ACT::Vector{FloatingPoint}, GRA::Vector{FloatingPoint})
	@inbounds begin
		# Get maximum net value.
		max::FloatingPoint = NET[1]
		for i = 2:length(NET)
			if NET[i] > max
				max = NET[i]
			end
		end
		# Exponentiate and compute sum.
		sum::FloatingPoint = 0.0
		for i = 1:length(NET)
			ACT[i] = exp(NET[i] - max)
			sum += ACT[i]
		end
		# Normalize.
		if sum > 0.0
			for i = 1:length(ACT)
				ACT[i] /= sum
			end
		end
		# Set gradient!!!
		#
		#
	end
end

function activation_tanh!(NET::Vector{FloatingPoint}, ACT::Vector{FloatingPoint}, GRA::Vector{FloatingPoint})
	@inbounds begin
		for i = 1:length(NET)
			ACT[i] = tanh(NET[i])
			GRA[i] = 1.0 - (ACT[i] * ACT[i])
		end
	end
end

### Error functions.

function error_squared!(YH::Vector{FloatingPoint}, Y::Vector{FloatingPoint}, OUT::Vector{FloatingPoint})
	@inbounds begin
		for i = 1:length(Y)
			OUT[i] = (YH - Y).^2
		end
	end
end