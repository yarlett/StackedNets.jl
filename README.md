# DeepNets

[![Build Status](https://travis-ci.org/yarlett/DeepNets.jl.svg?branch=master)](https://travis-ci.org/yarlett/DeepNets.jl)

DeepNets aims to provide a simple interface to "deep" stacks of neural network units that can be trained with gradient descent of selected error measures.

DeepNets can consist of any number of Layers stacked on top of one another. Each layer can have a different number of output units that feed into the next layer above in the stack, and can have different activation functions.

Currently supported activation functions include exponential, linear, rectified linear, sigmoidal, softmax, softplus, and tanh activations.

Currently supported error functions are squared error and cross entropy.

Support for dropout of hidden activations during training will also be added in the future.

## Logistic Regression Example

```julia
logistic_spec = [(5, ""), (1, "sigmoid")]
logistic = DeepNet{Float64}(logistic_spec, "cross_entropy")
```

## Classifying MNIST Digits Example

Let's say you want to construct a DeepNet consisting of 50 input units, connected to 100 sigmoidal units, connected to 50 tanh units, connected to 30 linear (output) units. And let's say you want to train the network using the cross entropy loss function. This can be accomplished with:

```julia
spec = [(50, ""), (100, "sigmoid"), (50, "tanh"), (30, "linear")]
DN = DeepNet{Float64}(spec, "cross_entropy")
```

Now let's say you want to compute the gradient of the error function with respect to every parameter in the net. That's also easy:

```julia
X = randn(1000, 50) # Create 1000 input cases.
Y = rand(1000, 30)  # Create 1000 output cases.
gradient_update_batch(X, Y, DN)
```

L.GB and L.GW within each Layer in the DeepNet stack will now contain the required gradient information for its corresponding parameter.