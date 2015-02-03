immutable DeepNet{T<:FloatingPoint}
	layers::Vector{Layer{T}}
	function DeepNet(spec::Array{(Int, ASCIIString), 1})
		if length(spec) <= 1
			return error("DeepNet specification is too short.")
		end
		if minimum([u for (u, a) in spec]) <= 0
			return error("Invalid number of units in DeepNet specification.")
		end
		layers = Array(Layer{T}, length(spec)-1)
		for l = 2:length(spec)
			units1, activation1 = spec[l-1]
			units2, activation2 = spec[l]
			layers[l-1] = Layer{T}(units1, units2, activation2)
		end
	end
end

function forward(X::Vector, DN::DeepNet)
	DN.layers[1].O[:] = X[:]
	for l = 2:length(DN.layers)
		forward(DN.layers[l-1].O, DN.layers[l])
	end
end