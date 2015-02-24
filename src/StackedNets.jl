module StackedNets

	export error!,
	forward!,
	gradient_check,
	gradient_update!,
	train_sgd!,
	StackedNet,
	Units

	include("activation_functions.jl")
	include("layer.jl")
	include("stackednet.jl")
	include("error_functions.jl")
	include("units.jl")
	include("utils.jl")
end