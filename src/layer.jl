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
	DACT_DNET::Matrix{T}
	DELTA::Matrix{T}
	E::Matrix{T}
	DE_DYH::Matrix{T}
	activation::ASCIIString
	activation_function!::Function

	function Layer(ni::Int, no::Int, activation_type::ASCIIString; scale::T=1e-3)
		if ni > 0 && no > 0
			# Initialize weights and biases for layer.
			W, B = get_layer_parameters(ni, no, scale=scale)
			# Initialize storage for gradient information.
			GW, GB = zeros(W), zeros(B)
			# Initialize storage for upper level units.
			NET = zeros(B)
			ACT = zeros(B)
			DACT_DNET = zeros(B)
			DELTA = zeros(B)
			E = zeros(B)
			DE_DYH = zeros(B)
			# Set activation function for layer.
			activation, activation_function! = activation_function_selector(activation_type)
			# Create and return the object.
			new(ni, no, W, B, GW, GB, NET, ACT, DACT_DNET, DELTA, E, DE_DYH, activation, activation_function!)
		else
			error("Invalid number of units used to initialize Layer object (ni=$ni; no=$no) to create Layer object.")
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
			# Apply nonlinearity and set gradient information related to nonlinearity.
			L.activation_function!(L.NET, L.ACT, L.DACT_DNET)
		end
	end
end