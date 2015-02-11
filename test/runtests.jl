using DeepNets
using Base.Test

# Function to numerically check the analytic error gradients.
function test_deepnet_gradient(; step=1e-8, tol=1e-5)
	# Construct a deep network.
	units = _generate_random_units()
	DN = DeepNet{Float64}(units, error_type="squared_error")
	# Construct an input / output pair.
	X = rand(units[1].n)
	Y = rand(units[end].n)
	# Set gradient information on the pattern and compare it to numerically derived gradients.
	gradient_update(DN, X, Y)
	for l = 1:length(DN.layers)
		L = DN.layers[l]
		# Compared error gradients for weights with numercially derived versions.
		for i = 1:size(L.W, 1)
			for j = 1:size(L.W, 2)
				curw = L.W[i, j]
				L.W[i, j] = curw - step
				EN = error(DN, X, Y)
				L.W[i, j] = curw + step
				EP = error(DN, X, Y)
				numerical_gradient = (EP - EN) / (2.0 * step)
				@test_approx_eq_eps(L.GW[i, j], numerical_gradient, tol)
				L.W[i, j] = curw
			end
		end
		# Compared error gradients for biases with numercially derived versions.
		for b = 1:length(L.B)
			curb = L.B[b]
			L.B[b] = curb - step
			EN = error(DN, X, Y)
			L.B[b] = curb + step
			EP = error(DN, X, Y)
			numerical_gradient = (EP - EN) / (2.0 * step)
			@test_approx_eq_eps(L.GB[b], numerical_gradient, tol)
			L.B[b] = curb
		end
	end
end

# Function to check forward propagation through nets works.
function test_deepnet_batch()
	# Create our data (10000 cases).
	X = rand(10, 10000)
	Y = rand(3, 10000)
	# Create DeepNet.
	units = [Units(10), Units(50, activation_type="sigmoid"), Units(50, activation_type="sigmoid"), Units(3, activation_type="linear")]
	DN = DeepNet{Float64}(units, error_type="squared_error")
	# Compute the gradient on the whole batch.
	gradient_update(DN, X, Y)
end

# Generates random arrays of units for testing purposes.
function _generate_random_units()
	activations = ["exponential", "linear", "rectified_linear", "sigmoid", "softmax", "softplus", "tanh"]
	units = Units[Units(rand(2:20))]
	for u = 1:rand(1:20)
		push!(units, Units(rand(2:20), activation_type=activations[rand(1:length(activations))]))
	end
	push!(units, Units(5, activation_type="sigmoid"))
	units
end

# Run tests.
@time test_deepnet_gradient()
@time test_deepnet_batch()