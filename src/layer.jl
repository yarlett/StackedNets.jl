include("units.jl")

immutable Layer{T<:FloatingPoint}
	ni::Int
	no::Int
	W::Matrix{T}
	B::Matrix{T}
	GW::Matrix{T}
	GB::Matrix{T}
	NET::Matrix{T}
	ACT::Matrix{T}
	DE_DNET::Matrix{T}
	DELTA::Matrix{T}
	E::Matrix{T}
	DE_DYH::Matrix{T}
	activation::ASCIIString
	activation_function!::Function
	activation_jacobian::Function

	function Layer(ni::Int, no::Int, activation::ASCIIString; scale::T=1e-3)
		if ni > 0 && no > 0
			# Initialize weights and biases for layer.
			W, B = get_layer_parameters(ni, no, scale=scale)
			# Initialize storage for gradient information.
			GW, GB = zeros(W), zeros(B)
			# Initialize storage for upper level units.
			NET = zeros(B)
			ACT = zeros(B)
			DE_DNET = zeros(B)
			DELTA = zeros(T, (ni, 1))
			E = zeros(B)
			DE_DYH = zeros(B)
			# Set activation function for layer.
			activation, activation_function!, activation_jacobian = activation_function_selector(activation)
			# Create and return the object.
			new(ni, no, W, B, GW, GB, NET, ACT, DE_DNET, DELTA, E, DE_DYH, activation, activation_function!, activation_jacobian)
		else
			error("Invalid number of units used to initialize Layer object (ni=$ni; no=$no) to create Layer object.")
		end
	end
end

# Propagate deltas from the layer above back through the layer.
function backward{T<:FloatingPoint}(L::Layer{T}, DELTA::Matrix{T}, p::Int)
	@inbounds begin
		# Set DE_DNET.
		for o = 1:L.no
			L.DE_DNET[o] = 0.0
			for oo = 1:L.no
				daoo_dnoo = L.activation_jacobian(o, oo, L.NET[:, 1], L.ACT[:, 1])
				L.DE_DNET[o, 1] += DELTA[oo] * daoo_dnoo
			end
		end
		# Set deltas.
		for i = 1:L.ni
			L.DELTA[i] = 0.0
			for o = 1:L.no
				L.DELTA[i] += L.DE_DNET[o, 1] * L.W[i, o]
			end
		end
	end
end

# Propagate a pattern forward through a layer.
function forward{T<:FloatingPoint}(L::Layer{T}, IN::Matrix{T}, p::Int)
	@inbounds begin
		for o = 1:L.no
			# Calculate net values.
			L.NET[o] = L.B[o]
			for i = 1:L.ni
				L.NET[o] += IN[i, p] * L.W[i, o]
			end
		end
		# Set activations and gradient information related to activations.
		L.activation_function!(L.NET, L.ACT)
	end
end