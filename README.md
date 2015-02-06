# DeepNets

[![Build Status](https://travis-ci.org/yarlett/DeepNets.jl.svg?branch=master)](https://travis-ci.org/yarlett/DeepNets.jl)

DeepNets aims to provide a simple interface to "deep" stacks of neural network units that can be trained with gradient descent of selected error measures.

DeepNets can consist of any number of layers stacked on top of one another. Each layer can have a diffferent number of output units that feed into the next layer above in the stack, and can have different activation functions.

Currently supported activation functions include exponential, linear, rectified linear, sigmoidal, softmax, and tanh activations.

Currently only the squared error function is supported, although other error functions will be added and supported in the future. Support for dropout of hidden activations will also be added in the future.