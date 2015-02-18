using DeepNets
using MNIST

# Uses DeepNets to train a logistic regression model to classify MNIST handwritten digits.

function digits_to_indicators(digits)
	digit_indicators = zeros(Float64, (10, length(digits)))
	for j = 1:length(digits)
		digit_indicators[int(digits[j])+1, j] = 1.0
	end
	digit_indicators
end

# Load MNIST training and testing data.
XTR, YTR = traindata()
YTR = digits_to_indicators(YTR)
XTE, YTE = testdata()
YTE = digits_to_indicators(YTE)
println("Training data: X has size $(size(XTR)); Y has size $(size(YTR)).")
println("Testing  data: X has size $(size(XTE)); Y has size $(size(YTE)).")
println()

# Define multinomial logistic classifier model in DeepNets.
units = [Units(size(XTR, 1)), Units(10, activation_type="softmax")]
deepnet = DeepNet{Float64}(units, error_type="cross_entropy")
println("Intial training error is $(error(deepnet, XTR, YTR)).")
println()

# Define a custom error function (non-differentiable so we can't use it in learning).
function error_percent(YH::Matrix{Float64}, Y::Matrix{Float64})
	error = 0.0
	for j = 1:size(YH, 2)
		_, i1 = findmax(YH[:, j])
		_, i2 = findmax(Y[:, j])
		if i1 != i2
			error += 1.0
		end
	end
	error = 100.0 * (error / size(YH, 2))
	error
end

# Perform stochastic gradient descent on the deep net.
@time df = train_sgd(
	deepnet,
	XTR,
	YTR,
	X_testing=XTE,
	Y_testing=YTE,
	custom_loss=error_percent,
	iterations=10000,
	iterations_report=1000,
	learning_rate=1e-5,
	minibatch_size=100,
	minibatch_replace=true,
	report=true
)
println()
println(df)