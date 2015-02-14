using DataFrames
using StatsBase

immutable DeepNet{T<:FloatingPoint}
	layers::Vector{Layer{T}}
	error_type::ASCIIString
	error_function!::Function

	function DeepNet(units::Vector{Units}; error_type::ASCIIString="squared_error", scale::T=1e-3)
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
			layers[u] = Layer{T}(units1.n, units2.n, units2.activation_type, scale=scale)
		end
		# Set error function.
		error_type, error_function! = error_function_selector(error_type)
		# Create and return the object.
		new(layers, error_type, error_function!)
	end
end

# Returns the error on a specific pattern.
function error{T<:FloatingPoint}(DN::DeepNet{T}, X::Matrix{T}, Y::Matrix{T}, p::Int)
	forward(DN, X, p)
	L = DN.layers[end]
	DN.error_function!(L.ACT, Y[:, p:p], L.E, L.DE_DYH)
	sum(L.E)
end

function error{T<:FloatingPoint}(DN::DeepNet{T}, X::Matrix{T}, Y::Matrix{T})
	@inbounds begin
		E::T = 0.0
		for p = 1:size(X, 2)
			E += error(DN, X, Y, p)
		end
	end
	E
end

# Forward propagate a pattern through a DeepNet.
function forward{T<:FloatingPoint}(DN::DeepNet{T}, X::Matrix{T}, p::Int)
	@inbounds begin
		forward(DN.layers[1], X, p)
		for l = 2:length(DN.layers)
			forward(DN.layers[l], DN.layers[l - 1].ACT, 1)
		end
	end
end

# Returns output activations in a DeepNet for a set of input patterns.
function forward{T<:FloatingPoint}(DN::DeepNet{T}, X::Matrix{T})
	@inbounds begin
		no = DN.layers[end].no
		np = size(X, 2)
		Y = zeros(T, (no, np))
		for p = 1:np
			forward(DN, X, p)
			Y[:, p] = DN.layers[end].ACT
		end
	end
	Y
end

# Numerically checks the analytic gradient of a DeepNet and returns a data frame of results.
function gradient_check{T<:FloatingPoint}(DN::DeepNet{T}, X::Matrix{T}, Y::Matrix{T}; step=1e-6, tolerance=1e-6)
	# Initialize data frame.
	df = DataFrame(layer=Int64[], parameter=Int64[], analytic_grad=T[], numerical_grad=T[], absolute_error=T[], ok=Bool[])
	# Set the gradient information.
	# Iterate over layers, and parameters within each layer.
	for l = 1:length(DN.layers)
		for (M, GM) in ((DN.layers[l].W, DN.layers[l].GW), (DN.layers[l].B, DN.layers[l].GB))
			for i = 1:length(M)
				current_parameter_value = M[i]
				# Get the analytic gradient.
				gradient_reset(DN)
				gradient_update(DN, X, Y)
				analytic_gradient = GM[i]
				# Get the numerical gradient.
				M[i] = current_parameter_value - step
				EN = error(DN, X, Y)
				M[i] = current_parameter_value + step
				EP = error(DN, X, Y)
				numerical_gradient = (EP - EN) / (2.0 * step)
				M[i] = current_parameter_value
				# Compare gradient values.
				absolute_error = abs(analytic_gradient - numerical_gradient)
				push!(df, (l, i, analytic_gradient, numerical_gradient, absolute_error, absolute_error < tolerance? true : false))
			end
		end
	end
	df
end

# Finds the maximum absolute gradient value.
function gradient_maxabs{T<:FloatingPoint}(DN::DeepNet{T})
	@inbounds begin
		gmax::T = -Inf
		for l = 1:length(DN.layers)
			for M in (DN.layers[l].GW, DN.layers[l].GB)
				thismaxabs = maximum(abs(M))
				if thismaxabs > gmax
					gmax = thismaxabs
				end
			end
		end
	end
	gmax
end

# Zeros out all the gradient information in a DeepNet.
function gradient_reset{T<:FloatingPoint}(DN::DeepNet{T})
	@inbounds begin
		for l = 1:length(DN.layers)
			L = DN.layers[l]
			for i = 1:length(L.GW)
				L.GW[i] = 0.0
			end
			for i = 1:length(L.GB)
				L.GB[i] = 0.0
			end
		end
	end
end

# Increment the gradient information (GW and GB) on each layer based on a single input-output pattern.
function gradient_update{T<:FloatingPoint}(DN::DeepNet{T}, X::Matrix{T}, Y::Matrix{T}, p::Int)
	@inbounds begin
		# Forward propagate the input pattern through the network.
		forward(DN, X, p)
		# Backpropagate the deltas for each unit in the network.
		for l = length(DN.layers):-1:1
			L = DN.layers[l]
			# Set deltas for output units.
			if l == length(DN.layers)
				DN.error_function!(L.ACT, Y[:, p:p], L.E, L.DELTA)
				for o = 1:L.no
					L.DELTA[o] *= L.DACT_DNET[o]
				end
			else
				Lup = DN.layers[l+1]
				for i = 1:Lup.ni
					L.DELTA[i] = 0.0
					for o = 1:Lup.no
						L.DELTA[i] += Lup.DELTA[o] * Lup.W[i, o]
					end
					L.DELTA[i] *= L.DACT_DNET[i]
				end
			end
			# Increment the gradient information.
			for o = 1:L.no
				if l == 1
					for i = 1:L.ni
						L.GW[i, o] += X[i, p] * L.DELTA[o]
					end
				else
					for i = 1:L.ni
						L.GW[i, o] += DN.layers[l - 1].ACT[i, 1] * L.DELTA[o]
					end
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
			gradient_update(DN, X, Y, p)
		end
	end
end

# Update the parameters of a DeepNet based on the accumulated gradient information.
function parameters_update{T<:FloatingPoint}(DN::DeepNet{T}, lr::T; reset_gradient=true)
	@inbounds begin
		for l = 1:length(DN.layers)
			L = DN.layers[l]
			for (M, GM) in ((L.W, L.GW), (L.B, L.GB))
				for i = 1:length(M)
					M[i] -= lr * GM[i]
					if reset_gradient
						GM[i] = 0.0
					end
				end
			end
		end
	end
end

function train_sgd{T<:FloatingPoint}(DN::DeepNet{T}, X::Matrix{T}, Y::Matrix{T}; X_testing=false, Y_testing=false, iterations::Int=1000, iterations_report::Int=100, learning_rate::T=1e-2, minibatch_size::Int=100, minibatch_replace::Bool=true, report::Bool=true)
	@inbounds begin
		num_patterns::Int64 = size(X, 2)
		# Minibatch size cannot be larger than number of patterns when sampling without replacement.
		if (minibatch_size > num_patterns) && !minibatch_replace
			minibatch_size = num_patterns
		end
		# Reserve space for minibatch integers.
		minibatch_ints = zeros(Int, minibatch_size)
		minibatch_domain = 1:num_patterns
		# Adjust the learning rate to account for the size of the minibatch.
		learning_rate_use = learning_rate / T(minibatch_size)
		# Create results dataframe if reporting is required.
		if report
			if X_testing !== false
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
				gradient_update(DN, X, Y, minibatch_int)
			end
			# Update the parameters based on the gradient information.
			parameters_update(DN, learning_rate_use, reset_gradient=true)
			# Decide whether to record performance.
			if report && (iteration % iterations_report == 0)
				if X_testing !== false
					row = (iteration, error(DN, X, Y), error(DN, X_testing, Y_testing))
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