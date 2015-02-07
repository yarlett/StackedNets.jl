### DeepNet type (a DeepNet is a stack of Layers).

immutable DeepNet{T<:FloatingPoint}
	# Data.
	layers::Vector{Layer{T}}
	error_type::ASCIIString
	error_function!::Function
	# Constructor.
	function DeepNet(spec::Array{(Int, ASCIIString), 1}, error_type::ASCIIString)
		if length(spec) < 2
			return error("DeepNet specification is too short.")
		end
		if minimum([num_units for (num_units, activation) in spec]) <= 0
			return error("Invalid number of units in DeepNet specification.")
		end
		layers = Array(Layer{T}, length(spec)-1)
		for l = 2:length(spec)
			units1, _ = spec[l-1]
			units2, activation2 = spec[l]
			layers[l-1] = Layer{T}(units1, units2, activation2)
		end
		# Set error function.
		error_type, error_function! = error_function_selector(error_type)
		# Create and return the object.
		new(layers, error_type, error_function!)
	end
end

function forward{T<:FloatingPoint}(X::Vector{T}, DN::DeepNet{T})
	forward(X, DN.layers[1])
	for l = 2:length(DN.layers)
		forward(DN.layers[l-1].ACT, DN.layers[l])
	end
end

### Gradient functions.

# Zeros out all the gradient information in a DeepNet.
function gradient_reset{T<:FloatingPoint}(DN::DeepNet{T})
	for l = 1:length(DN.layers)
		L = DN.layers[l]
		for i = 1:size(L.GW, 1)
			for j = 1:size(L.GW, 2)
				L.GW[i, j] = 0.0
			end
		end
		for b = 1:length(L.GB)
			L.GB[b] = 0.0
		end
	end
end

function gradient_update_batch{T<:FloatingPoint}(X::Matrix{T}, Y::Matrix{T}, DN::DeepNet{T})
	gradient_reset(DN)
	for i = 1:size(X, 1)
		x, y = vec(X[i, :]), vec(Y[i, :])
		gradient_update_pattern(x, y, DN)
	end
end

# Updates the gradient information in a DeepNet for a given input-output pair.
function gradient_update_pattern{T<:FloatingPoint}(X::Vector{T}, Y::Vector{T}, DN::DeepNet{T})
	# Forward propagate the input pattern through the network.
	forward(X, DN)
	# Compute the deltas for each unit.
	L = DN.layers[end]
	ERR = zeros(T, L.no)
	DN.error_function!(L.ACT, Y, ERR, L.DELTA)
	for o = 1:L.no
		L.DELTA[o] *= L.DACT_DNET[o]
	end
	for l in length(DN.layers)-1:-1:1
		L1, L2 = DN.layers[l], DN.layers[l+1]
		for i = 1:L2.ni
			L1.DELTA[i] = 0.0
			for o = 1:L2.no
				L1.DELTA[i] += L2.W[i, o] * L2.DELTA[o]
			end
			L1.DELTA[i] *= L1.DACT_DNET[i]
		end
	end
	# Now update gradient information.
	for l = 1:length(DN.layers)
		L = DN.layers[l]
		for i = 1:L.ni
			for o = 1:L.no
				L.GW[i, o] += L.IN[i] * L.DELTA[o]
			end
		end
		for o = 1:L.no
			L.GB[o] += L.DELTA[o]
		end
	end
end

function error{T<:FloatingPoint}(X::Vector{T}, Y::Vector{T}, DN::DeepNet{T})
	forward(X, DN)
	ERR = zeros(T, length(DN.layers[end].ACT))
	DE_DYH = zeros(T, length(DN.layers[end].ACT))
	L = DN.layers[end]
	DN.error_function!(L.ACT, Y, ERR, DE_DYH)
	sum(ERR)
end