using Base.BLAS
using DataFrames
using StatsBase

immutable StackedNet{T<:FloatingPoint}
	layers::Vector{Layer{T}}
	error::ASCIIString
	error_function!::Function
	error_function_prime!::Function

	function StackedNet(units::Vector{Units}; error::ASCIIString="squared_error")
		if length(units) < 2
			return error("StackedNet units specification is too short.")
		end
		if minimum([unit.n for unit in units]) <= 0
			return error("Invalid number of units in StackedNet units specification.")
		end
		# Iterate over sequential paits of units and construct the required layers.
		layers = Array(Layer{T}, length(units)-1)
		for u = 1:length(units)-1
			units1, units2 = units[u], units[u + 1]
			layers[u] = Layer{T}(units1.n, units2.n, units2.activation)
		end
		# Set error function.
		error, error_function!, error_function_prime! = error_function_selector(error)
		# Create and return the object.
		new(layers, error, error_function!, error_function_prime!)
	end
end

# Returns the patternwise error on a specific pattern.
function error!{T<:FloatingPoint}(DN::StackedNet{T}, X::Matrix{T}, Y::Matrix{T}, p::Int)
	forward!(DN, X, p)
	L = DN.layers[end]
	y = sub(Y, :, p)
	DN.error_function!(L.ACT, y, L.E)
	sum(L.E)
end

# Returns the patternwise error on a set of patterns.
function error!{T<:FloatingPoint}(DN::StackedNet{T}, X::Matrix{T}, Y::Matrix{T})
	@inbounds begin
		E::T = 0.0
		for p = 1:size(X, 2)
			E += error!(DN, X, Y, p)
		end
		E = E / size(X, 2)
	end
	E
end

# Forward propagate a pattern through a StackedNet.
function forward!{T<:FloatingPoint}(DN::StackedNet{T}, X::Matrix{T}, p::Int)
	@inbounds begin
		x = sub(X, :, p)
		forward!(DN.layers[1], x)
		# forward!(DN.layers[1], X, p)
		for l = 2:length(DN.layers)
			forward!(DN.layers[l], DN.layers[l - 1].ACT)
		end
	end
end

# Returns output activations in a StackedNet for a set of input patterns.
function forward!{T<:FloatingPoint}(DN::StackedNet{T}, X::Matrix{T})
	@inbounds begin
		no = DN.layers[end].no
		np = size(X, 2)
		Y = zeros(T, (no, np))
		for p = 1:np
			forward!(DN, X, p)
			for o = 1:no
				Y[o, p] = DN.layers[end].ACT[o]
			end
		end
	end
	Y
end

# Numerically checks the analytic gradient of a StackedNet and returns a data frame of results.
function gradient_check{T<:FloatingPoint}(DN::StackedNet{T}, X::Matrix{T}, Y::Matrix{T}; step=1e-8, tolerance=1e-5)
	# Initialize data frame.
	df = DataFrame(layer=Int64[], parameter=Int64[], analytic_grad=T[], numerical_grad=T[], absolute_error=T[], ok=Bool[])
	# Iterate over layers, and parameters within each layer.
	for l = 1:length(DN.layers)
		for (M, GM) in ((DN.layers[l].W, DN.layers[l].GW), (DN.layers[l].B, DN.layers[l].GB))
			for i = 1:length(M)
				current_parameter_value = M[i]
				# Set the gradient information.
				gradient_reset!(DN)
				gradient_update!(DN, X, Y)
				# Get the analytic gradient (scale it to reflect patternwise gradient).
				analytic_gradient = GM[i] / size(X, 2)
				# Get the numerical gradient.
				M[i] = current_parameter_value - step
				EN = error!(DN, X, Y)
				M[i] = current_parameter_value + step
				EP = error!(DN, X, Y)
				numerical_gradient = (EP - EN) / (2.0 * step)
				M[i] = current_parameter_value
				# Compare gradient values.
				absolute_error = abs(analytic_gradient - numerical_gradient)
				# absolute_error = abs(analytic_gradient - numerical_gradient)
				push!(df, (l, i, analytic_gradient, numerical_gradient, absolute_error, absolute_error < tolerance? true : false))
			end
		end
	end
	df
end

# Finds the maximum absolute gradient value.
function gradient_maxabs{T<:FloatingPoint}(DN::StackedNet{T})
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

# Zeros out all the gradient information in a StackedNet.
function gradient_reset!{T<:FloatingPoint}(DN::StackedNet{T})
	@inbounds begin
		for l = 1:length(DN.layers)
			L = DN.layers[l]
			fill!(L.GW, 0.0)
			fill!(L.GB, 0.0)
		end
	end
end

# Increment the gradient information (GW and GB) on each layer based on a single input-output pattern.
function gradient_update!{T<:FloatingPoint}(DN::StackedNet{T}, X::Matrix{T}, Y::Matrix{T}, p::Int)
	@inbounds begin
		# Forward propagate the input pattern through the network.
		forward!(DN, X, p)
		# Walk backward through the stacked network.
		nl = length(DN.layers)
		for l = nl:-1:1
			# Set references to layers.
			Lup = l == nl ? nothing : DN.layers[l + 1]
			L = DN.layers[l]
			Ldn = l == 1 ? nothing : DN.layers[l - 1]
			# Backpropagation.
			if l == nl
				y = sub(Y, :, p)
				DN.error_function_prime!(L.ACT, y, L.DE_DYH)
				backward!(L, L.DE_DYH)
			else
				backward!(L, Lup.DELTAS)
			end
			# Increment gradient information.
			axpy!(L.no, 1.0, L.DE_DNET, 1, L.GB, 1)
			if l == 1
				x = sub(X, :, p)
				ger!(1.0, X[:, p], L.DE_DNET, L.GW)
			else
				ger!(1.0, Ldn.ACT, L.DE_DNET, L.GW)
			end
		end
	end
end

# Increment the gradient information (GW and GB) on each layer based on a set of input-output pairs.
function gradient_update!{T<:FloatingPoint}(DN::StackedNet{T}, X::Matrix{T}, Y::Matrix{T})
	@inbounds begin
		for p = 1:size(X, 2)
			gradient_update!(DN, X, Y, p)
		end
	end
end

# Update the parameters of a StackedNet based on the accumulated gradient information.
function parameters_update!{T<:FloatingPoint}(DN::StackedNet{T}, lr::T; reset_gradient::Bool=true)
	@inbounds begin
		for l = 1:length(DN.layers)
			L = DN.layers[l]
			axpy!(length(L.W), -lr, L.GW, 1, L.W, 1)
			axpy!(length(L.B), -lr, L.GB, 1, L.B, 1)
			if reset_gradient
				fill!(L.GW, 0.0)
				fill!(L.GB, 0.0)
			end
		end
	end
end

function get_learning_rate{T<:FloatingPoint}(DN::StackedNet{T}, max_parameter_change::T)
	gmax::T = -Inf
	for l = 1:length(DN.layers)
		wmax = maximum(abs(DN.layers[l].GW))
		if wmax > gmax
			gmax = wmax
		end
		bmax = maximum(abs(DN.layers[l].GB))
		if bmax > gmax
			gmax = bmax
		end
	end
	max_parameter_change / gmax
end

function train_sgd!{T<:FloatingPoint}(DN::StackedNet{T}, X_training::Matrix{T}, Y_training::Matrix{T}; X_testing::Union(Nothing, Matrix{T})=nothing, Y_testing::Union(Nothing, Matrix{T})=nothing, custom_error::Union(Nothing, Function)=nothing, iterations::Int=1000, iterations_report::Int=100, learning_rate::T=1e-2, minibatch_size::Int=100, minibatch_replace::Bool=true)
	@inbounds begin
		num_patterns::Int64 = size(X_training, 2)
		# Minibatch size cannot be larger than number of patterns when sampling without replacement.
		if (minibatch_size > num_patterns) && !minibatch_replace
			minibatch_size = num_patterns
		end
		# Reserve space for minibatch integers.
		minibatch_ints = zeros(Int, minibatch_size)
		minibatch_domain = 1:num_patterns
		# # Adjust the learning rate to account for the size of the minibatch.
		# learning_rate_use = learning_rate / minibatch_size
		# Create results dataframe if reporting is required.
		if custom_error == nothing
			df = DataFrame(iteration=Int64[], error_training=T[], error_testing=T[])
		else
			df = DataFrame(iteration=Int64[], error_training=T[], error_testing=T[], custom_error_training=T[], custom_error_testing=T[])
		end
		# Perform the required number of iterations of learning.
		gradient_reset!(DN)
		for iteration = 1:iterations
			# Increment the gradient information based on the minibatch.
			sort!(sample!(minibatch_domain, minibatch_ints, replace=minibatch_replace))
			for minibatch_int in minibatch_ints
				gradient_update!(DN, X_training, Y_training, minibatch_int)
			end
			# Set the learning rate such that the maximum parameter update has the desired value.
			learning_rate_use = get_learning_rate(DN, learning_rate)
			# Update the parameters based on the gradient information.
			parameters_update!(DN, learning_rate_use, reset_gradient=true)
			# Decide whether to record performance.
			if (iteration % iterations_report) == 0
				# Construct row to add to dataframe.
				if custom_error == nothing
					row = (iteration, error!(DN, X_training, Y_training), error!(DN, X_testing, Y_testing))
				else
					YH_training = forward!(DN, X_training)
					YH_testing = forward!(DN, X_testing)
					row = (iteration, error!(DN, X_training, Y_training), error!(DN, X_testing, Y_testing), custom_error(YH_training, Y_training), custom_error(YH_testing, Y_testing))
				end
				push!(df, row)
				println("Iteration $iteration. Error $(df[end, :error_training]).")
			end
		end
	end
	df
end