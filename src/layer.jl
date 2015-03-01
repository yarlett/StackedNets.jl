using Base.BLAS

include("units.jl")

immutable Layer{T<:FloatingPoint}
	ni::Int
	no::Int
	W::Matrix{T}
	B::Vector{T}
	GW::Matrix{T}
	GB::Vector{T}
	NET::Vector{T}
	ACT::Vector{T}
	DE_DNET::Vector{T}
	E::Vector{T}
	DE_DYH::Vector{T}
	DELTAS::Vector{T}
	activation::ASCIIString
	activation_function!::Function
	activation_jacobian::Function
	activation_full_jacobian::Bool

	function Layer(ni::Int, no::Int, activation::ASCIIString; scale::T=1e-3)
		if ni > 0 && no > 0
			# Initialize weights and biases for layer.
			W, B = get_layer_parameters(ni, no, scale=scale)
			# Initialize storage for gradient information.
			GW, GB = zeros(W), zeros(B)
			# Initialize storage for upper level units.
			NET = zeros(T, no)
			ACT = zeros(T, no)
			DE_DNET = zeros(T, no)
			E = zeros(T, no)
			DE_DYH = zeros(T, no)
			DELTAS = zeros(T, ni)
			# Set activation function for layer.
			activation, activation_function!, activation_jacobian, activation_full_jacobian = activation_function_selector(activation)
			# Create and return the object.
			new(ni, no, W, B, GW, GB, NET, ACT, DE_DNET, E, DE_DYH, DELTAS, activation, activation_function!, activation_jacobian, activation_full_jacobian)
		else
			error("Invalid number of units used to initialize Layer object (ni=$ni; no=$no) to create Layer object.")
		end
	end
end

# Propagate delta quantities coming from the layer above downward through the layer.
function backward!{T<:FloatingPoint}(L::Layer{T}, DELTAS::Vector{T})
	@inbounds begin
		# Set DE_DNET on the layer.
		if L.activation_full_jacobian
			for o = 1:L.no
				L.DE_DNET[o] = 0.0
				for oo = 1:L.no
					L.DE_DNET[o] += DELTAS[oo] * L.activation_jacobian(o, oo, L.NET, L.ACT)
				end
			end
		else
			for o = 1:L.no
				L.DE_DNET[o] = DELTAS[o] * L.activation_jacobian(o, o, L.NET, L.ACT)
			end
		end
		# Set DELTAS on the layer.
		L.DELTAS[:] = 0.0
		gemv!('N', 1.0, L.W, L.DE_DNET, 1.0, L.DELTAS)
	end
end

# Propagate an incoming pattern forward through a layer (vector input).
function forward!{T<:FloatingPoint}(L::Layer{T}, IN::Vector{T})
	@inbounds begin
		# Calculate net values.
		blascopy!(L.no, L.B, 1, L.NET, 1)
		gemv!('T', 1.0, L.W, IN, 1.0, L.NET)
		# Set activations and gradient information related to activations.
		L.activation_function!(L.NET, L.ACT)
	end
end

# Propagate an incoming pattern forward through a layer (matrix input plus pattern index).
function forward!{T<:FloatingPoint}(L::Layer{T}, IN::Matrix{T}, p::Int)
	@inbounds begin
		# Calculate net values.
		blascopy!(L.no, L.B, 1, L.NET, 1)
		gemv!('T', 1.0, L.W, IN[:, p], 1.0, L.NET)
		# Set activations and gradient information related to activations.
		L.activation_function!(L.NET, L.ACT)
	end
end