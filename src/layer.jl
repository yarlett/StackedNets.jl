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
	E::Matrix{T}
	DE_DYH::Matrix{T}
	DELTAS::Matrix{T}
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
			NET = zeros(T, (no, 1))
			ACT = zeros(T, (no, 1))
			DE_DNET = zeros(T, (no, 1))
			E = zeros(T, (no, 1))
			DE_DYH = zeros(T, (no, 1))
			DELTAS = zeros(T, (ni, 1))
			# Set activation function for layer.
			activation, activation_function!, activation_jacobian = activation_function_selector(activation)
			# Create and return the object.
			new(ni, no, W, B, GW, GB, NET, ACT, DE_DNET, E, DE_DYH, DELTAS, activation, activation_function!, activation_jacobian)
		else
			error("Invalid number of units used to initialize Layer object (ni=$ni; no=$no) to create Layer object.")
		end
	end
end

# Propagate delta quantities coming from the layer above downward through the layer.
function backward!{T<:FloatingPoint}(L::Layer{T}, DELTAS::Matrix{T})
	@inbounds begin
		# Set DE_DNET.
		jacobian = L.activation_jacobian
		for o = 1:L.no
			L.DE_DNET[o] = 0.0
			for oo = 1:L.no
				L.DE_DNET[o] += DELTAS[oo, 1] * jacobian(o, oo, L.NET[:, 1], L.ACT[:, 1])
			end
		end
		# Set DELTAS.
		for i = 1:L.ni
			L.DELTAS[i] = 0.0
			for o = 1:L.no
				L.DELTAS[i] += L.DE_DNET[o] * L.W[i, o]
			end
		end
	end
end

# Propagate an incoming pattern forward through a layer.
function forward!{T<:FloatingPoint}(L::Layer{T}, IN::Matrix{T}, p::Int)
	@inbounds begin
		# Calculate net values.
		for o = 1:L.no
			L.NET[o] = L.B[o]
			for i = 1:L.ni
				L.NET[o] += IN[i, p] * L.W[i, o]
			end
		end
		# Set activations and gradient information related to activations.
		L.activation_function!(L.NET, L.ACT)
	end
end