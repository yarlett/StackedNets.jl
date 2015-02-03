# Activation functions.

activation_linear(x::FloatingPoint) = x

function rectified_linear(x::FloatingPoint)
	if x > 0.0
		return x
	else
		return 1e-3 * x
	end
end

activation_sigmoid(x::FloatingPoint) = 1.0 / (1.0 + exp(-x))

activation_tanh(x::FloatingPoint) = tanh(x)

# Layer type.

immutable Layer{T<:FloatingPoint}
	ni::Int
	no::Int
	W::Matrix{T}
	B::Vector{T}
	Onet::Vector{T}
	O::Vector{T}
	activation_function::Function
	function Layer(ni::Int, no::Int)
		if ni > 0 && no > 0
			W = zeros(T, (ni, no))
			for i = 1:size(W, 1)
				for j = 1:size(W, 2)
					W[i, j] = 1e-3 * randn()
				end
			end
			B = zeros(T, no)
			Onet = zeros(T, no)
			O = zeros(T, no)
			new(ni, no, W, B, Onet, O, activation_sigmoid)
		else
			error("Invalid input dimensions to create Layer object.")
		end
	end
end

function forward(X::Vector, L::Layer)
	for o = 1:L.no
		# Calculate net values.
		L.Onet[o] = L.B[o]
		for i = 1:L.ni
			L.Onet[o] += (X[i] * L.W[i, o])
		end
		# Apply nonlinearity.
		L.O[o] = L.activation_function(L.Onet[o])
	end
end