using DeepNets
using Base.Test

# Function to numerically check the analytic error gradients.
function test_deepnet_gradient(; step=1e-8, tol=1e-5)
	# Construct a deep network.
	spec = _generate_random_deepnet_spec()
	DN = DeepNet{Float64}(spec, "squared_error")
	# Construct an input / output pair.
	X = rand(spec[1][1])
	Y = rand(spec[end][1])
	# Set gradient information on the pattern and compare it to numerically derived gradients.
	gradient_update_pattern(X, Y, DN)
	for l = 1:length(DN.layers)
		L = DN.layers[l]
		# Compared error gradients for weights with numercially derived versions.
		for i = 1:size(L.W, 1)
			for j = 1:size(L.W, 2)
				curw = L.W[i, j]
				L.W[i, j] = curw - step
				EN = error(X, Y, DN)
				L.W[i, j] = curw + step
				EP = error(X, Y, DN)
				numerical_gradient = (EP - EN) / (2.0 * step)
				@test_approx_eq_eps(L.GW[i, j], numerical_gradient, tol)
				L.W[i, j] = curw
			end
		end
		# Compared error gradients for biases with numercially derived versions.
		for b = 1:length(L.B)
			curb = L.B[b]
			L.B[b] = curb - step
			EN = error(X, Y, DN)
			L.B[b] = curb + step
			EP = error(X, Y, DN)
			numerical_gradient = (EP - EN) / (2.0 * step)
			@test_approx_eq_eps(L.GB[b], numerical_gradient, tol)
			L.B[b] = curb
		end
	end
end

# Function to check forward propagation through nets works.
function test_deepnet_batch()
	# Create our data.
	X = rand(10000, 10)
	Y = rand(10000, 3)
	# Create DeepNet.
	spec = [(10, ""), (50, "sigmoid"), (50, "sigmoid"), (3, "linear")]
	DN = DeepNet{Float64}(spec, "squared_error")
	# Compute the gradient on the whole batch.
	gradient_update_batch(X, Y, DN)
end

# Generates random DeepNet specifications to test.
function _generate_random_deepnet_spec()
	num_layers = rand(2:20)
	activations = ["exponential", "linear", "rectified_linear", "sigmoid", "softmax", "tanh"]
	spec = [(rand(2:20), "")]
	for l = 1:num_layers
		push!(spec, (rand(2:20), activations[rand(1:length(activations))]))
	end
	push!(spec, (5, "sigmoid"))
	spec
end

# Run tests.
@time test_deepnet_gradient()
@time test_deepnet_batch()