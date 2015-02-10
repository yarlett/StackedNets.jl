using StatsBase

### DeepNet type (a DeepNet is a stack of Layers).
include("units.jl")

immutable DeepNet{T<:FloatingPoint}
	# Data.
	layers::Vector{Layer{T}}
	error_type::ASCIIString
	error_function!::Function
	# Constructor.
	function DeepNet(units::Vector{Units}, error_type::ASCIIString)
		if length(units) < 2
			return error("DeepNet units specification is too short.")
		end
		if minimum([unit.n for unit in units]) <= 0
			return error("Invalid number of units in DeepNet units specification.")
		end
		# Iterate over sequential paits of units and construct the required layers.
		layers = Array(Layer{T}, length(units)-1)
		for u = 1:length(units)-1
			units1, units2 = units[u], units[u + 1]
			layers[u] = Layer{T}(units1.n, units2.n, units2.activation_type)
		end
		# Set error function.
		error_type, error_function! = error_function_selector(error_type)
		# Create and return the object.
		new(layers, error_type, error_function!)
	end
end

function error{T<:FloatingPoint}(DN::DeepNet{T}, X::Vector{T}, Y::Vector{T})
	forward(DN, X)
	ERR = zeros(T, length(DN.layers[end].ACT))
	DE_DYH = zeros(T, length(DN.layers[end].ACT))
	L = DN.layers[end]
	DN.error_function!(L.ACT, Y, ERR, DE_DYH)
	sum(ERR)
end

function error{T<:FloatingPoint}(DN::DeepNet{T}, X::Matrix{T}, Y::Matrix{T})
	E::T = 0.0
	ERR = zeros(T, length(DN.layers[end].ACT))
	DE_DYH = zeros(T, length(DN.layers[end].ACT))
	L = DN.layers[end]
	for p = 1:size(X, 2)
		forward(DN, X[:, p])
		DN.error_function!(L.ACT, Y[:, p], ERR, DE_DYH)
		E += sum(ERR)
	end
	E / size(X, 2)
end

function forward{T<:FloatingPoint}(DN::DeepNet{T}, X::Vector{T})
	forward(X, DN.layers[1])
	for l = 2:length(DN.layers)
		forward(DN.layers[l-1].ACT, DN.layers[l])
	end
end

function forward{T<:FloatingPoint}(DN::DeepNet{T}, X::Matrix{T})
	no = DN.layers[end].no
	Y = zeros(T, (no, size(X, 2)))
	for p = 1:size(X, 2)
		forward(DN, X[:, p])
		for o = 1:no
			Y[o, p] = DN.layers[end].ACT[o]
		end
	end
	Y
end

# Zeros out all the gradient information in a DeepNet.
function gradient_reset{T<:FloatingPoint}(DN::DeepNet{T})
	for l = 1:length(DN.layers)
		L = DN.layers[l]
		for i = 1:size(L.GW, 1)
			for j = 1:size(L.GW, 2)
				L.GW[i, j] = 0.0
			end
		end
		for b = 1:length(L.GB)
			L.GB[b] = 0.0
		end
	end
end

function gradient_update{T<:FloatingPoint}(DN::DeepNet{T}, X::Matrix{T}, Y::Matrix{T})
	gradient_reset(DN)
	for p = 1:size(X, 2)
		gradient_update(DN, X[:, p], Y[:, p])
	end
end

function gradient_update{T<:FloatingPoint}(DN::DeepNet{T}, X::Vector{T}, Y::Vector{T})
	# Forward propagate the input pattern through the network.
	forward(DN, X)
	# Compute the deltas for each unit.
	L = DN.layers[end]
	ERR = zeros(T, L.no)
	DN.error_function!(L.ACT, Y, ERR, L.DELTA)
	for o = 1:L.no
		L.DELTA[o] *= L.DACT_DNET[o]
	end
	for l in length(DN.layers)-1:-1:1
		L1, L2 = DN.layers[l], DN.layers[l+1]
		for i = 1:L2.ni
			L1.DELTA[i] = 0.0
			for o = 1:L2.no
				L1.DELTA[i] += L2.W[i, o] * L2.DELTA[o]
			end
			L1.DELTA[i] *= L1.DACT_DNET[i]
		end
	end
	# Now update gradient information.
	for l = 1:length(DN.layers)
		L = DN.layers[l]
		for i = 1:L.ni
			for o = 1:L.no
				L.GW[i, o] += L.IN[i] * L.DELTA[o]
			end
		end
		for o = 1:L.no
			L.GB[o] += L.DELTA[o]
		end
	end
end

function update_parameters{T<:FloatingPoint}(DN::DeepNet{T}, lr::T; zero_gradient=true)
	for l = 1:length(DN.layers)
		L = DN.layers[l]
		for i = 1:size(L.W, 1)
			for j = 1:size(L.W, 2)
				L.W[i, j] -= lr * L.GW[i, j]
				if zero_gradient
					L.GW[i, j] = 0.0
				end
			end
		end
		for b = 1:length(L.B)
			L.B[b] -= lr * L.GB[b]
			if zero_gradient
				L.B[b] = 0.0
			end
		end
	end
end

function train_sgd{T<:FloatingPoint}(DN::DeepNet{T}, X, Y; its=1000, lr::T=1e-2, mbsize=100, mbreplace=true)
	if (mbsize > size(X, 2)) && !mbreplace
		mbsize = size(X, 2)
	end

	uselr = lr / mbsize

	for it = 1:its
		# Sample minibatch.
		mbints = sort(sample(1:size(X, 2), mbsize, replace=mbreplace))
		mbx, mby = X[:, mbints], Y[:, mbints]
		# Reset gradient and set gradient based on minibatch.
		gradient_update(DN, mbx, mby)
		# Update the parameters based on the gradient information.
		update_parameters(DN, uselr; zero_gradient=true)
	end
end