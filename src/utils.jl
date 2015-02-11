function get_layer_parameters{T<:FloatingPoint}(ni::Int, no::Int; scale::T=1e-3)
	# Initialize weight matrix.
	W = zeros(T, (ni, no))
	for i = 1:size(W, 1)
		for j = 1:size(W, 2)
			W[i, j] = scale * randn()
		end
	end
	# Initialize bias units.
	B = zeros(T, no)
	for i = 1:length(B)
		B[i] += scale * randn()
	end
	# Return.
	W, B
end

function minibatch_assign!{T<:FloatingPoint}(MMB::Matrix{T}, M::Matrix{T}, mcols::Vector{Int})
	@inbounds begin
		I = size(M, 1)
		for j = 1:length(mcols)
			mcol = mcols[j]
			for i = 1:I
				MMB[i, j] = M[i, mcol]
			end
		end
	end
end