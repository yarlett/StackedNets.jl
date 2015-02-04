module DeepNets

	export forward,
	DeepNet

	include("layer.jl")
	include("deepnet.jl")
	include("nonlinearities.jl")
end