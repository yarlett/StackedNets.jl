### Layer type.

immutable Layer{T<:FloatingPoint}
	# Data.
	ni::Int
	no::Int
	IN::Vector{T}
	W::Matrix{T}
	B::Vector{T}
	NET::Vector{T}
	ACT::Vector{T}
	DACT_DNET::Vector{T}
	DELTA::Vector{T}
	GW::Matrix{T}
	GB::Vector{T}
	activation_function!::Function
	# Constructor.
	function Layer(ni::Int, no::Int, activation_type::AbstractString)
		if ni > 0 && no > 0
			IN = zeros(T, ni)
			# Initialize weights and biases for layer.
			W = zeros(T, (ni, no))
			for i = 1:size(W, 1)
				for j = 1:size(W, 2)
					W[i, j] = 1e-3 * randn()
				end
			end
			B = zeros(T, no)
			# Initialize storage for upper level units.
			NET = zeros(T, no)
			ACT = zeros(T, no)
			DACT_DNET = zeros(T, no)
			DELTA = zeros(T, no)
			GW = zeros(T, (ni, no))
			GB = zeros(T, no)
			# Set activation function for layer.
			activation_function! = activation_function_selector(activation_type)
			# Create and return the object.
			new(ni, no, IN, W, B, NET, ACT, DACT_DNET, DELTA, GW, GB, activation_function!)
		else
			error("Invalid number of units used to initialize Layer object (ni=$ni; no=$no) to create Layer object.")
		end
	end
end

function forward(IN::Vector, L::Layer)
	@inbounds for i = 1:L.ni
		L.IN[i] = IN[i]
	end
	@inbounds for o = 1:L.no
		# Calculate net values.
		L.NET[o] = L.B[o]
		for i = 1:L.ni
			L.NET[o] += (L.IN[i] * L.W[i, o])
		end
		# Apply nonlinearity and set gradient information related to nonlinearity.
		L.activation_function!(L.NET, L.ACT, L.DACT_DNET)
	end
end
