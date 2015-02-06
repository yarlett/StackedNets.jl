module DeepNets

	export backward,
	error,
	forward,
	DeepNet

	include("layer.jl")
	include("deepnet.jl")
	include("nonlinearities.jl")
end