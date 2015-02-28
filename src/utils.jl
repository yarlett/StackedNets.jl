function get_layer_parameters{T<:FloatingPoint}(ni::Int, no::Int; scale::T=1e-3)
	# Initialize weight matrix.
	W = zeros(T, (ni, no))
	for i = 1:length(W)
		W[i] = scale * randn()
	end
	# Initialize bias units.
	B = zeros(T, no)
	for i = 1:length(B)
		B[i] += scale * randn()
	end
	# Return.
	W, B
end

function clip{T<:FloatingPoint}(val::T; low::T=1e-10, high::T=1.0-1e-10)
	if val < low
		return low
	elseif val > high
		return high
	else
		return val
	end
end