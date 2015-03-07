using Base.BLAS

include("units.jl")

immutable Layer{T<:FloatingPoint}
	ni::Int
	no::Int
	W::Matrix{T}
	B::Vector{T}
	GW::Matrix{T}
	GB::Vector{T}
	UW::Matrix{T}
	UB::Vector{T}
	NET::Vector{T}
	ACT::Vector{T}
	DE_DNET::Vector{T}
	E::Vector{T}
	DE_DYH::Vector{T}
	DELTAS::Vector{T}
	activation::ASCIIString
	activation_function!::Function
	activation_backward!::Function

	function Layer(ni::Int, no::Int, activation::ASCIIString)
		if ni > 0 && no > 0
			# Initialize weights and biases for layer.
			sigma = 1e-2
			W = zeros(T, (ni, no))
			for i = 1:length(W)
				W[i] = sigma * randn()
			end
			B = zeros(T, no)
			for i = 1:length(B)
				B[i] = sigma * randn()
			end
			# Initialize storage for gradient information.
			GW, GB = zeros(W), zeros(B)
			UW, UB = zeros(W), zeros(B)
			# Initialize storage for upper level units.
			NET = zeros(T, no)
			ACT = zeros(T, no)
			DE_DNET = zeros(T, no)
			E = zeros(T, no)
			DE_DYH = zeros(T, no)
			DELTAS = zeros(T, ni)
			# Set activation function for layer.
			activation, activation_function!, activation_backward! = activation_function_selector(activation)
			# Create and return the object.
			new(ni, no, W, B, GW, GB, UW, UB, NET, ACT, DE_DNET, E, DE_DYH, DELTAS, activation, activation_function!, activation_backward!)
		else
			error("Invalid number of units used to initialize Layer object (ni=$ni; no=$no) to create Layer object.")
		end
	end
end

# Propagate delta quantities coming from the layer above downward through the layer.
function backward!{T<:FloatingPoint}(L::Layer{T}, DELTAS_ABOVE::Vector{T})
	@inbounds begin
		# Set DE_DNET on the layer.
		L.activation_backward!(L.NET, L.ACT, DELTAS_ABOVE, L.DE_DNET)
		# Set DELTAS on the layer.
		fill!(L.DELTAS, 0.0)
		gemv!('N', 1.0, L.W, L.DE_DNET, 1.0, L.DELTAS)
	end
end

# Propagate an incoming pattern forward through a layer.
function forward!{T<:FloatingPoint}(L::Layer{T}, IN::AbstractVector{T})
	@inbounds begin
		# Calculate net values.
		blascopy!(L.no, L.B, 1, L.NET, 1)
		gemv!('T', 1.0, L.W, IN, 1.0, L.NET)
		# Set activations and gradient information related to activations.
		L.activation_function!(L.NET, L.ACT)
	end
end