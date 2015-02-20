module DeepNets

	export error,
	forward,
	gradient_check,
	gradient_update,
	train_sgd,
	DeepNet,
	Units

	include("activation_functions.jl")
	include("layer.jl")
	include("deepnet.jl")
	include("error_functions.jl")
	include("units.jl")
	include("utils.jl")
end