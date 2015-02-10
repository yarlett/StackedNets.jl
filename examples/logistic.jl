using DeepNets
using MNIST

# This script uses DeepNets to train a logistic regression model to classify MNIST handwritten digits as being odd or even.

function is_even(digits)
	for i = 1:length(digits)
		digit = int(digits[i])
		digits[i] = iseven(digit) ? 1.0 : 0.0
	end
	digits
end

# Load MNIST training data.
xtr, ytr = traindata()
ytr = is_even(ytr)
ytr = reshape(ytr, (1, length(ytr)))
xte, yte = testdata()
yte = is_even(yte)
yte = reshape(yte, (1, length(yte)))

# Define logistic classifier model.
logistic_units = [Units(size(xtr, 1)), Units(1, activation_type="sigmoid")]
logistic_deepnet = DeepNet{Float64}(logistic_units, "cross_entropy")

# Perform stochastic gradient descent.
xent_tr = error(logistic_deepnet, xtr, ytr)
xent_te = error(logistic_deepnet, xte, yte)
println("Training cross entropy = $xent_tr. Testing cross entropy = $xent_te.")

@time train_sgd(logistic_deepnet, xtr, ytr, its=10000, lr=1e-6, mbsize=100, mbreplace=true)

# Measure training and testing peformance.
yhtr = forward(logistic_deepnet, xtr)
yhte = forward(logistic_deepnet, xte)
acc_tr = mean((ytr .> 0.5) .== (yhtr .> 0.5))
acc_te = mean((yte .> 0.5) .== (yhte .> 0.5))
println("Training classification accuracy = $acc_tr. Testing classification accuracy = $acc_te.")
