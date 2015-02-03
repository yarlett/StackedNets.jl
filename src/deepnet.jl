immutable DeepNet{T<:FloatingPoint}
	layers::Vector{Layer{T}}
	function DeepNet(spec::Array{(Int, AbstractString), 1})
		if length(spec) <= 1
			return error("DeepNet specification is too short.")
		end
		if minimum([u for (u, a) in spec]) <= 0
			return error("Invalid number of units in DeepNet specification.")
		end
		layers = Array(Layer{T}, length(spec))
		for l = 1:length(spec)-1
			units1, activation1 = spec[l]
			units2, activation2 = spec[l+1]
			layers[l] = Layer{T}(units1, units2, activation1)
		end
	end
end
