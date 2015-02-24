using StackedNets
using MNIST

# Uses StackedNets to train a logistic regression model, and a feedforward network with  to classify MNIST handwritten digits.

function digits_to_indicators(digits)
	digit_indicators = zeros(Float64, (10, length(digits)))
	for j = 1:length(digits)
		digit_indicators[int(digits[j])+1, j] = 1.0
	end
	digit_indicators
end

# Define a custom error function (non-differentiable so we can't optimize it during learning).
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

# Load MNIST training and testing data.
XTR, YTR = traindata()
XTR ./= 255.0
YTR = digits_to_indicators(YTR)
XTE, YTE = testdata()
XTE ./= 255.0
YTE = digits_to_indicators(YTE)
println("Training data: X has size $(size(XTR)); Y has size $(size(YTR)).")
println("Testing  data: X has size $(size(XTE)); Y has size $(size(YTE)).")
println()

# Define multinomial logistic classifier model in StackedNets.
units1 = [Units(size(XTR, 1)), Units(10, activation="softmax")]
deepnet1 = StackedNet{Float64}(units1, error="cross_entropy")

# Define feedforward neural network model with rectified linear units in StackedNets.
units2 = [Units(size(XTR, 1)), Units(25, activation="rectified_linear"),  Units(10, activation="softmax")]
deepnet2 = StackedNet{Float64}(units2, error="cross_entropy")

# Train the 2 models.
for (label, deepnet) in (("Multinomial Logistic", deepnet1), ("784-100-10 Rectified Linear", deepnet2))
	# Report initial training error.
	println("Intial training error of $label model is $(error!(deepnet2, XTR, YTR)).")
	println()

	# Perform stochastic gradient descent on the deep net (training results returned as a DataFrame).
	@time df = train_sgd!(
		deepnet,
		XTR,
		YTR,
		X_testing=XTE,
		Y_testing=YTE,
		custom_error=error_percent,
		iterations=100000,
		iterations_report=1000,
		learning_rate=1e-2,
		minibatch_size=100,
		minibatch_replace=true,
	)
	println(df)
	println()

	# Report final error.
	YH = forward!(deepnet, XTR)
	tr_error = error_percent(YH, YTR)
	YH = forward!(deepnet, XTE)
	te_error = error_percent(YH, YTE)
	println("Final test classification error for $label model on MNIST is: training $(tr_error)%; testing $(te_error)%.")
	println()
end