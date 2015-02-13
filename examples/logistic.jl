using DeepNets
using MNIST

# Uses DeepNets to train a logistic regression model to classify MNIST handwritten digits as being odd or even.

function is_even(digits)
	for i = 1:length(digits)
		digit = int(digits[i])
		digits[i] = iseven(digit) ? 1.0 : 0.0
	end
	digits
end

# Load MNIST training and testing data.
xtr, ytr = traindata()
ytr = is_even(ytr)
ytr = reshape(ytr, (1, length(ytr)))
xte, yte = testdata()
yte = is_even(yte)
yte = reshape(yte, (1, length(yte)))
println("Training data: X is $(size(xtr, 1)) by $(size(xtr, 2)); Y is $(size(ytr, 1)) by $(size(ytr, 2)).")
println("Testing data : X is $(size(xte, 1)) by $(size(xte, 2)); Y is $(size(yte, 1)) by $(size(yte, 2)).")
println()

# Define logistic classifier model in DeepNets (logistic classifier has 784 input units and 1 sigmoid output unit).
logistic_units = [Units(size(xtr, 1)), Units(1, activation_type="sigmoid")]
logistic_deepnet = DeepNet{Float64}(logistic_units, error_type="cross_entropy")

# Perform stochastic gradient descent.
@time df = train_sgd(logistic_deepnet, xtr, ytr, XTEST=xte, YTEST=yte, iterations=10000, iterations_report=1000, learning_rate=1e-6, minibatch_size=200, minibatch_replace=true, report=true)
println()
println(df)
println()

# Measure training and testing performance.
yhtr = forward(logistic_deepnet, xtr)
yhte = forward(logistic_deepnet, xte)
acc_tr = mean((ytr .> 0.5) .== (yhtr .> 0.5))
acc_te = mean((yte .> 0.5) .== (yhte .> 0.5))
println("Training classification accuracy = $acc_tr. Testing classification accuracy = $acc_te.")
println()