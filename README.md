#StackedNets

[![Build Status](https://travis-ci.org/yarlett/StackedNets.jl.svg?branch=master)](https://travis-ci.org/yarlett/StackedNets.jl)

StackedNets provides a simple interface to "deep" stacks of neural network units that can be trained using gradient descent over defined error measures.

StackedNets can consist of any number of Layers stacked on top of one another. Each Layer can have a different number of output units that feed into the next layer above in the stack, and can have different activation functions.

Currently supported activation functions include exponential, linear, rectified linear, sigmoid, softmax, softplus, and tanh activations. StackedNets is able to compute non-diagonal Jacobian terms when it computes the differentials of unit activations with respect to unit inputs, and incorporates these values in its backpropagation algorithm, which means it can model activation functions where the activation of a unit depends on the input to units other than the unit in question.

Currently supported error functions are squared error and cross entropy.

StackedNets has been designed to make it relatively easy to write and add new activation and error functions to the source code. Over time it may become possible to pass in custom functions at the call-level.

Support for dropout of visible/hidden activations during training will also be added in the future, as will the ability to save/load trained models.

The main priority in StackedNets so far has been to develop a flexible and clean API to specify StackedNets, train them, and use them for prediction. Hopefully StackedNets is reasonably performant for a CPU-based framework (it calls BLAS routines where possible); adding GPU compatibility may be a target for future development.

##Specifying StackedNets

StackedNets are constructed by first specifying a Units list. A Units list specifies the number of units in each layer of a StackedNet. For example, if we wanted to specify a model with 10 input units, feeding through to 3 output units with sigmoid activations, then we would define our Units list as follows

```julia
using StackedNets
units = [Units(10), Units(3, activation="sigmoid")]
```

Note that the first Units object in the list always corresponds to the input layer and defaults to having linear activations when no activation type is specified (it is typically desirable to have linear activations in the input layer of a network).

Alternatively, we can specify a more complicated network in the following way

```julia
units = [Units(100), Units(100, activation="sigmoid"), Units(50, activation="rectified_linear"), Units(10, activation="softmax")]
```

##Constructing StackedNets

StackedNets themselves are constructed from Units lists as follows

```julia
stackednet = StackedNet{Float64}(units, error="cross_entropy")
```

The Float64 type specifies the type of the inputs, parameters, and outputs used by the StackedNet, and must be a sub-class of Julia's FloatingPoint type (Float32 would be the other primary use-case I would imagine, but you never know). The error keyword specifies the error function used to compute output-target errors during training.

##Training StackedNets

Right now StackedNets can be trained using stochastic gradient descent, where minibatches can be randomly sampled from a training set (either with or without replacement).

##Classifying MNIST Digits

For a more fully worked out example, check out [this one](examples/mnist_classification.jl). This script specifies 2 models -- a multinomial logistic classifier, and a more complex feedforward network with hidden units -- and trains them to classify handwritten digits into the 10 digit classes in the MNIST data set.
