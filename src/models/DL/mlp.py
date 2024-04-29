"""
Module for the Multi-Layer Perceptron (MLP) regressor (Session 10).

"""
from collections.abc import Sequence
from typing import Optional

import numpy as np

from base import BaseRegressor


class ModularLinearLayer:
    """
    Modular linear layer of a neural network with L2 regularization term and bias.
    This layer has its own backward and update function, which are needed for use in a modular MLP.

    Attributes:
        in_features: Number of input features
        out_features: Number of output features
        weight_: Weight matrix of shape (in_features, out_features)
        bias_: Bias vector of shape (out_features,)
        _weight_grad: Gradient of the weight matrix
        _bias_grad: Gradient of the bias vector
        _prev_input: Input to the layer, used for backpropagation
    """

    def __init__(self, in_features: int, out_features: int, rng: np.random.RandomState = np.random.RandomState()):
        self._bias_grad = None
        self._weight_grad = None
        self.in_features = in_features
        self.out_features = out_features
        scale = np.sqrt(6.0 / (in_features + out_features))
        self.weight_ = rng.uniform(-scale, scale, size=(in_features, out_features))
        self.bias_ = rng.uniform(-scale, scale, out_features)
        self._prev_input = None

    def __call__(self, X: np.array) -> np.array:
        """
        Forward pass of the layer
        Parameters
        ----------
        X : np.array
            Input to the layer of shape (batch_size, in_features)

        Returns
        -------
        np.array
            Output of the layer of shape (batch_size, out_features)

        """
        self._prev_input = X.dot(self.weight_) + self.bias_
        return self._prev_input

    def backward(self, upstream_grad: np.ndarray, alpha: Optional[float] = None) -> np.ndarray:
        """
        Backward pass of the layer - only needed for Modular MLPs.
        Parameters
        ----------
        upstream_grad : np.ndarray
            Gradient of the loss w.r.t. the output of the layer
        alpha : Optional[float]
            Regularization parameter

        Returns
        -------
        np.ndarray
            Gradient of the loss w.r.t. the input of the layer

        """
        # TODO Implement backward pass
        raise NotImplementedError
        # derivative of Cost w.r.t weight

        # Add regularization terms for weights

        # derivative of Cost w.r.t bias, sum across rows

        # derivative of Cost w.r.t _prev_input

    def update(self, lr: float) -> None:
        """
        Update the parameters of the layer - only needed for Modular MLPs.

        Parameters
        ----------
        lr : float
            Learning rate for the update
        """


class TanhLayer:
    """
    Layer for tanh activation function.
    """

    def __init__(self):
        self._prev_result = None

    def __call__(self, X):
        self._prev_result = np.tanh(X)
        return self._prev_result

    def backward(self, upstream_grad):
        """
        Compute the gradient of the ReLU activation function.

        Parameters
        ----------
        upstream_grad : np.ndarray
            The gradient of the cost function w.r.t. the output of the ReLU layer.

        Returns
        -------
        np.ndarray
            The gradient of the cost function w.r.t. the input of the ReLU layer.

        """
        # TODO Implement backward pass
        raise NotImplementedError


class MLPRegressor(BaseRegressor):
    """
    Multi-layer perceptron regressor with modular layer definition, configurable activation and L2 regularization.

    Parameters
    ----------
    hidden_layer_sizes : Sequence[int]
        Number of neurons in each hidden layer
    lr : float
        Learning rate
    epochs : int
        Number of epochs to train for
    random_state : Optional[int]
        Random seed for layer initialization
    alpha : float
        Regularization parameter
    activation : Optional[str]
        Activation function to use. Can be 'relu', 'tanh' or 'sigmoid'
    """

    def __init__(self, hidden_layer_sizes: Sequence[int] = (3, 5, 3), lr: float = 0.01, epochs: int = 100,
                 random_state: Optional[int] = None, alpha: float = 0.0001, activation: Optional[str] = 'tanh'):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.lr = lr
        self.epochs = epochs
        self.layers_ = None
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.alpha = alpha
        self.activation = activation

    def fit(self, X: np.array, y: np.array):
        in_size = X.shape[1]

        self.layers_ = []
        for size in self.hidden_layer_sizes:
            self.layers_.append(ModularLinearLayer(in_size, size, self.rng))
            in_size = size

            if self.activation == 'tanh':
                self.layers_.append(TanhLayer())

        self.layers_.append(ModularLinearLayer(in_size, 1, self.rng))

        # Gradient descent. For each batch...
        for epoch in range(0, self.epochs):
            # TODO Implement gradient descent

            # Forward pass

            # Calculate loss and its gradient

            # Backpropagation of gradients

            # Update weights and biases in each layer according to their gradients

            raise NotImplementedError

        return self
