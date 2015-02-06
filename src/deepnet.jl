### DeepNet type (a DeepNet is a stack of Layers).

immutable DeepNet{T<:FloatingPoint}
	# Data.
	layers::Vector{Layer{T}}
	error_function!::Function
	# Constructor.
	function DeepNet(spec::Array{(Int, ASCIIString), 1})
		if length(spec) < 2
			return error("DeepNet specification is too short.")
		end
		if minimum([num_units for (num_units, activation) in spec]) <= 0
			return error("Invalid number of units in DeepNet specification.")
		end
		layers = Array(Layer{T}, length(spec)-1)
		for l = 2:length(spec)
			units1, activation1 = spec[l-1]
			units2, activation2 = spec[l]
			layers[l-1] = Layer{T}(units1, units2, activation2)
		end
		# Create and return the object.
		new(layers, error_squared!)
	end
end

function forward(X::Vector, DN::DeepNet)
	forward(X, DN.layers[1])
	for l = 2:length(DN.layers)
		forward(DN.layers[l-1].ACT, DN.layers[l])
	end
end

function backward(X::Vector, Y::Vector, DN::DeepNet)
	# Forward propagate the input pattern through the network.
	forward(X, DN)
	# Compute the deltas for each unit.
	L = DN.layers[end]
	for o = 1:L.no
		L.DELTA[o] = (L.ACT[o] - Y[o]) * L.DACT_DNET[o] # Assuming squared error function.
	end
	for l in length(DN.layers)-1:-1:1
		L1, L2 = DN.layers[l], DN.layers[l+1]
		for o1 = 1:size(L2.W, 1)
			L1.DELTA[o1] = 0.0
			for o2 = 1:size(L2.W, 2)
				L1.DELTA[o1] += L2.W[o1, o2] * L2.DELTA[o2]
			end
			L1.DELTA[o1] *= L1.DACT_DNET[o1]
		end
	end
	# Now update gradient information.
	for l = 1:length(DN.layers)
		L = DN.layers[l]
		for o1 = 1:size(L.W, 1)
			for o2 = 1:size(L.W, 2)
				L.GW[o1, o2] += L.IN[o1] * L.DELTA[o2]
			end
		end
		for o2 = 1:size(L.W, 2)
			L.GB[o2] += L.DELTA[o2]
		end
	end
end

function error{T<:FloatingPoint}(X::Vector{T}, Y::Vector{T}, DN::DeepNet{T})
	forward(X, DN)
	ERRS = zeros(T, length(DN.layers[end].ACT))
	error_squared!(DN.layers[end].ACT, Y, ERRS)
	E = sum(ERRS)
	E
end