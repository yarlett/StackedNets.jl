using DataFrames
using StatsBase

include("utils.jl")

### DeepNet type.
immutable DeepNet{T<:FloatingPoint}
	layers::Vector{Layer{T}}
	error_type::ASCIIString
	error_function!::Function

	function DeepNet(units::Vector{Units}; error_type::ASCIIString="squared_error")
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

# Returns patternwise error on a single input-output pair.
function error{T<:FloatingPoint}(DN::DeepNet{T}, X::Vector{T}, Y::Vector{T})
	forward(DN, X)
	ERR = zeros(T, length(DN.layers[end].ACT))
	DE_DYH = zeros(T, length(DN.layers[end].ACT))
	L = DN.layers[end]
	DN.error_function!(L.ACT, Y, ERR, DE_DYH)
	sum(ERR)
end

# Returns patternwise error on a set of input-output pairs.
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

# Sets activations in a DeepNet based on a single input pattern.
function forward{T<:FloatingPoint}(DN::DeepNet{T}, X::Vector{T})
	forward(X, DN.layers[1])
	for l = 2:length(DN.layers)
		forward(DN.layers[l-1].ACT, DN.layers[l])
	end
end

# Returns output activations in a DeepNet for a set of input patterns.
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

# Increment the gradient information (GW and GB) on each layer based on a single input-output pair.
function gradient_update{T<:FloatingPoint}(DN::DeepNet{T}, X::Vector{T}, Y::Vector{T})
	@inbounds begin
		# Forward propagate the input pattern through the network.
		forward(DN, X)
		# Backpropagate the deltas for each unit in the network.
		for l = length(DN.layers):-1:1
			L = DN.layers[l]
			# Set deltas for output units.
			if l == length(DN.layers)
				DN.error_function!(L.ACT, Y, L.ERR, L.DELTA)
				for o = 1:L.no
					L.DELTA[o] *= L.DACT_DNET[o]
				end
			else
				Lup = DN.layers[l+1]
				for i = 1:Lup.ni
					L.DELTA[i] = 0.0
					for o = 1:Lup.no
						L.DELTA[i] += Lup.W[i, o] * Lup.DELTA[o]
					end
					L.DELTA[i] *= L.DACT_DNET[i]
				end
			end
			# Update the gradient information.
			for o = 1:L.no
				for i = 1:L.ni
					L.GW[i, o] += L.IN[i] * L.DELTA[o]
				end
				L.GB[o] += L.DELTA[o]
			end
		end
	end
end

# Increment the gradient information (GW and GB) on each layer based on a set of input-output pairs.
function gradient_update{T<:FloatingPoint}(DN::DeepNet{T}, X::Matrix{T}, Y::Matrix{T})
	@inbounds begin
		for p = 1:size(X, 2)
			gradient_update(DN, X[:, p], Y[:, p])
		end
	end
end

# Update the parameters of a DeepNet based on the accumulated gradient information.
function parameters_update{T<:FloatingPoint}(DN::DeepNet{T}, lr::T; zero_gradient=true)
	@inbounds begin
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
end

function train_sgd{T<:FloatingPoint}(DN::DeepNet{T}, X::Matrix{T}, Y::Matrix{T}; XTEST=false, YTEST=false, iterations::Int=1000, iterations_report::Int=100, learning_rate::T=1e-2, minibatch_size::Int=100, minibatch_replace::Bool=true, report::Bool=true)
	@inbounds begin
		num_patterns::Int64 = size(X, 2)
		# Minibatch size cannot be larger than number of patterns when sampling without replacement.
		if (minibatch_size > num_patterns) && !minibatch_replace
			minibatch_size = num_patterns
		end
		# Scale learning rate to account for the size of the minibatch.
		learning_rate_use::T = learning_rate / T(minibatch_size)
		# Reserve space for minibatch vectors.
		minibatch_x = zeros(T, size(X, 1))
		minibatch_y = zeros(T, size(Y, 1))
		minibatch_ints = zeros(Int, minibatch_size)
		minibatch_domain = 1:num_patterns
		# Create results dataframe if reporting is required.
		if report
			if XTEST !== false
				df = DataFrame(iteration=Int64[], error_training=T[], error_testing=T[])
			else
				df = DataFrame(iteration=Int64[], error_training=T[])
			end
		end
		# Perform the required number of iterations of learning.
		gradient_reset(DN)
		for iteration = 1:iterations
			# Increment the gradient information based on the minibatch.
			sample!(minibatch_domain, minibatch_ints, replace=minibatch_replace)
			for minibatch_int in minibatch_ints
				for i = 1:size(X, 1)
					minibatch_x[i] = X[i, minibatch_int]
				end
				for i = 1:size(Y, 1)
					minibatch_y[i] = Y[i, minibatch_int]
				end
				gradient_update(DN, minibatch_x, minibatch_y)
			end
			# Update the parameters based on the gradient information.
			parameters_update(DN, learning_rate_use, zero_gradient=true)
			# Decide whether to record performance.
			if report && (iteration % iterations_report == 0)
				if XTEST !== false
					row = (iteration, error(DN, X, Y), error(DN, XTEST, YTEST))
				else
					row = (iteration, error(DN, X, Y))
				end
				push!(df, row)
				println("Iteration $iteration. Error $(df[end, :error_training]).")
			end
		end
		if report
			println()
		end
	end
	if report
		return df
	end
end