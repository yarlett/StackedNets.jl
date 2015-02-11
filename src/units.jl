### Units type.

immutable Units
	n::Int
	activation_type::ASCIIString
	activation_function!::Function

	function Units(n::Int; activation_type::ASCIIString="linear")
		if n > 0
			# Set activation function for layer.
			activation_type, activation_function! = activation_function_selector(activation_type)
			# Create and return the object.
			new(n, activation_type, activation_function!)
		else
			error("Invalid number of units used to initialize Unit object (n=$n).")
		end
	end
end
