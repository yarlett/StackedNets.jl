using DeepNets
using Base.Test

# Define test functions.
function test_deepnet_construction()
	#
	step = 1e-8
	tol = 1e-6
	# Construct a deep network.
	spec = generate_random_deepnet_spec()
	println(spec)
	DN = DeepNet{Float64}(spec)
	# Construct an input / output pair.
	X = rand(spec[1][1])
	Y = rand(spec[end][1])
	# Set gradient information on the pattern.
	backward(X, Y, DN)
	for l = 1:length(DN.layers)
		L = DN.layers[l]
		for i = 1:size(L.W, 1)
			for j = 1:size(L.W, 2)
				curw = L.W[i, j]
				E0 = error(X, Y, DN)
				L.W[i, j] += step
				E1 = error(X, Y, DN)
				g = (E1-E0) / step
				@test_approx_eq_eps(L.GW[i, j], g, tol)
				L.W[i, j] = curw
			end
		end
		for b = 1:length(L.B)
			curb = L.B[b]
			E0 = error(X, Y, DN)
			L.B[b] += step
			E1 = error(X, Y, DN)
			g = (E1 - E0) / step
			@test_approx_eq_eps(L.GB[b], g, tol)
			L.B[b] = curb
		end
	end
	true
end

function generate_random_deepnet_spec()
	num_layers = rand(2:20)
	#activations = ["softmax"]
	activations = ["exponential", "linear", "rectified_linear", "sigmoid", "tanh"]
	spec = [(rand(2:20), "")]
	for l = 1:num_layers
		push!(spec, (rand(2:20), activations[rand(1:length(activations))]))
	end
	spec
end

# Run tests.
@test test_deepnet_construction()