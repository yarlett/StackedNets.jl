immutable Units
	n::Int
	activation::ASCIIString
	activation_function!::Function

	function Units(n::Int; activation::ASCIIString="linear")
		if n > 0
			# Set activation function for layer.
			activation, activation_function! = activation_function_selector(activation)
			# Create and return the object.
			new(n, activation, activation_function!)
		else
			error("Invalid number of units used to initialize Unit object (n=$n).")
		end
	end
end
