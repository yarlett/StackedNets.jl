### Layer type.

immutable Layer{T<:FloatingPoint}
	# Data.
	ni::Int
	no::Int
	W::Matrix{T}
	B::Vector{T}
	NET::Vector{T}
	ACT::Vector{T}
	GRA::Vector{T}
	activation_function!::Function
	# Constructor.
	function Layer(ni::Int, no::Int, activation_type::AbstractString)
		if ni > 0 && no > 0
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
			GRA = zeros(T, no)
			# Set activation function for layer.
			activation_function! = activation_function_selector(activation_type)
			# Create and return the object.
			new(ni, no, W, B, Onet, O, activation_function!)
		else
			error("Invalid number of units used to initialize Layer object (ni=$ni; no=$no) to create Layer object.")
		end
	end
end

function forward(X::Vector, L::Layer)
	@inbounds for o = 1:L.no
		# Calculate net values.
		L.Onet[o] = L.B[o]
		for i = 1:L.ni
			L.Onet[o] += (X[i] * L.W[i, o])
		end
		# Apply nonlinearity and set gradient information.
		L.activation_function!(L.Onet, L.O, L.G)
	end
end

function backward()
end