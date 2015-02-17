function get_layer_parameters{T<:FloatingPoint}(ni::Int, no::Int; scale::T=1e-3)
	# Initialize weight matrix.
	W = zeros(T, (ni, no))
	for i = 1:size(W, 1)
		for j = 1:size(W, 2)
			W[i, j] = scale * randn()
		end
	end
	# Initialize bias units.
	B = zeros(T, (no, 1))
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