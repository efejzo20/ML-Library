"""
Module for the Convolutional Neural Network (CNN) classifier (Session 11).
"""

from typing import Optional, Tuple, Sequence, Union

import numpy as np

from numpy.lib.stride_tricks import as_strided
from numpy import einsum

from base import BaseClassifier
from src.mlp import ModularLinearLayer


class ConvLayer:
    """
    Convolutional layer of a convolutional neural network.

    Attributes:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the kernel
        stride: Stride of the convolution
        padding: Padding of the convolution
        bias_: Bias vector of shape (out_channels,)
        weight_: Weight matrix of shape (out_channels, in_channels, kernel_size, kernel_size)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int],
                 stride: Tuple[int, int] = (1, 1), padding: int = 0,
                 rng: np.random.RandomState = np.random.RandomState()):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias_ = rng.randn(out_channels, 1).squeeze()
        self.weight_ = rng.randn(*kernel_size, out_channels, in_channels)
        self._bias_grad = None
        self._weight_grad = None
        self._prev_input = None

    def __call__(self, X: np.array) -> np.array:
        """
        Forward pass of the layer
        Parameters
        ----------
        X : np.array
            Input to the layer of shape (batch_size, width, height, in_channels)

        Returns
        -------
        np.array
            Output of the layer of shape (batch_size, width_, height_, out_channels)

        """
        # TODO Implement forward pass
        raise NotImplementedError

    def backward(self, upstream_grad: np.ndarray, alpha: Optional[float] = None) -> np.ndarray:
        """
        Backward pass of the layer
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
        Update the parameters of the layer

        Parameters
        ----------
        lr : float
            Learning rate for the update
        """
        # TODO Implement weight update
        raise NotImplementedError


class PoolingLayer():
    """
    Pooling layer of a convolutional neural network.

    Attributes:
        kernel_size: Size of the kernel
        stride: Stride of the convolution
        padding: Padding of the convolution
    """

    def __init__(self, kernel_size: Tuple[int, int], stride: Tuple[int, int] = (1, 1), padding: int = 0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def __call__(self, X: np.array) -> np.array:
        """
        Forward pass of the layer
        Parameters
        ----------
        X : np.array
            Input to the layer of shape (batch_size, width, height, in_channels)

        Returns
        -------
        np.array
            Output of the layer of shape (batch_size, width, height, in_channels)

        """
        # TODO Implement forward pass
        raise NotImplementedError

    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        """
        Backward pass of the layer - only needed for Modular MLPs.
        Parameters
        ----------
        upstream_grad : np.ndarray
            Gradient of the loss w.r.t. the output of the layer

        Returns
        -------
        np.ndarray
            Gradient of the loss w.r.t. the input of the layer

        """
        # TODO Implement backward pass
        raise NotImplementedError


class SoftmaxLayer:
    def __init__(self):
        self._prev_result = None

    def __call__(self, X):
        # TODO Implement forward pass
        raise NotImplementedError

    def backward(self, upstream_grad):
        # TODO Implement backward pass
        raise NotImplementedError


class CNNClassifier(BaseClassifier):
    """
    Convolutional Neural Network Classifier

    Attributes:
        layers: Sequence of layers
        lr: Learning rate
        epochs: Number of epochs
        random_state: Random state
        alpha: Regularization parameter
        batch_size: Batch size
        out_layer_: Output layer
        softmax_layer_: Softmax layer
    """

    def __init__(self, layers: Sequence[Union[ConvLayer, PoolingLayer]], lr: float = 0.01, epochs: int = 1000,
                 random_state: Optional[int] = None, alpha: float = 0.0001, batch_size: int = 32):
        self.layers = layers
        self.lr = lr
        self.epochs = epochs
        self.random_state = random_state
        self.rng_ = np.random.RandomState(random_state)
        self.out_layer_: Optional[ModularLinearLayer] = None
        self.softmax_layer_: Optional[SoftmaxLayer] = None
        self.alpha = alpha
        self.batch_size = batch_size
