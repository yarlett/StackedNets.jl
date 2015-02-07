module DeepNets

	export error,
	forward,
	gradient_update_batch,
	gradient_update_pattern,
	DeepNet

	include("layer.jl")
	include("deepnet.jl")
	include("nonlinearities.jl")
end