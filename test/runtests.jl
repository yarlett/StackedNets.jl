using StackedNets
using Base.Test

# Function to numerically check the analytic error gradients.
function test_deepnet_gradient(; cases::Int=1)
	for activation in ["leaky_rectified_linear", "linear", "rectified_linear", "sigmoid", "softmax", "softplus", "tanh"]
		for error in ("absolute_error", "cross_entropy", "squared_error")
			println("Testing $activation units with $error errors.")
			# Construct a random deep network based on the activation and error types.
			units = _generate_random_unit_list(activation, error)
			DN = StackedNet{Float64}(units, error=error)
			# Check the layers in the net have the required activation type.
			for l = 1:length(DN.layers) - 1
				@test DN.layers[l].activation == activation
			end
			# Construct input / output cases.
			X = rand(units[1].n, cases)
			Y = rand(units[end].n, cases)
			# Set gradient information on the pattern and compare it to numerically derived gradients.
			results = gradient_check(DN, X, Y)
			ok, notok = sum(results[:ok]), sum(!results[:ok])
			if notok > 0
				println(results[!results[:ok], :])
			end
			println("$ok/$(ok+notok) gradient checks passed.")
			println()
			@test sum(!results[:ok]) == 0
		end
	end
end

# Generates random arrays of units for testing purposes.
function _generate_random_unit_list(activation, error)
	# Create a random number of layers of units with the required activation function.
	units = Units[Units(rand(2:20))]
	for u = 1:rand(1:10)
		push!(units, Units(rand(2:20), activation=activation))
	end
	# If error is cross_entropy, add a final layer to ensure outputs are probabilities.
	if error == "cross_entropy"
		push!(units, Units(rand(1:20), activation="softmax"))
	end
	units
end

# Run tests.
@time test_deepnet_gradient()