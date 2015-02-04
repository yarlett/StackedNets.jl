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
	DN.layers[1].O[:] = X[:]
	for l = 2:length(DN.layers)
		forward(DN.layers[l-1].O, DN.layers[l])
	end
end

