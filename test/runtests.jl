using DeepNets
using Base.Test

# Function to numerically check the analytic error gradients.
function test_deepnet_gradient()
	for error_type in ("squared_error", "cross_entropy")
		# Construct a deep network.
		units = _generate_random_units()
		DN = DeepNet{Float64}(units, error_type=error_type, scale=1e-1)
		# Construct an input / output pair.
		X = rand(units[1].n, 1)
		Y = rand(units[end].n, 1)
		# Set gradient information on the pattern and compare it to numerically derived gradients.
		results = gradient_check(DN, X, Y)
		ok, notok = sum(results[:ok]), sum(!results[:ok])
		println(results[!results[:ok], :])
		println("$ok/$(ok+notok) gradient checks passed.")
		@test sum(!results[:ok]) == 0
	end
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