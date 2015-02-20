using DeepNets
using Base.Test

# Function to numerically check the analytic error gradients.
function test_deepnet_gradient(; cases::Int=1)
	for error in ("squared_error", "cross_entropy")
		for activation in ["linear", "sigmoid", "softmax"]
			println("Testing $activation units with $error errors.")
			# Construct a deep network.

			units = [Units(2), Units(2, activation=activation)]
			# num_inputs = rand(1:100)
			# num_outputs = rand(1:20)
			# units = [Units(num_inputs)]
			# for i = rand(1:10)
			# 	push!(units, Units(rand(1:100), activation_type=activation_type))
			# end
			if error == "cross_entropy"
				push!(units, Units(5, activation="softmax"))
			end

			DN = DeepNet{Float64}(units, error=error, scale=1e-1)
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