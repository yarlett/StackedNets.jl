function clip{T<:FloatingPoint}(val::T; low::T=1e-10, high::T=1.0-1e-10)
	if val < low
		return low
	elseif val > high
		return high
	else
		return val
	end
end