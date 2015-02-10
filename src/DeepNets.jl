module DeepNets

	export error,
	forward,
	gradient_update,
	train_sgd,
	DeepNet,
	Units

	include("layer.jl")
	include("deepnet.jl")
	include("nonlinearities.jl")
	include("units.jl")
end