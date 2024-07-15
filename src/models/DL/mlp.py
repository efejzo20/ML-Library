from abc import ABC, abstractmethod
import numpy as np
from base import BaseClassifier
from typing import Optional, Tuple, Sequence, Union


class ModularLinearLayer:
    """
    Modular linear layer of a neural network with L2 regularization term and bias.
    This layer has its own backward and update function, which are needed for use in a modular MLP.
    They are not needed for the ANNClassifier and other NNs which compute gradients and updtes themselves.

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

        spread = np.sqrt(6.0 / (in_features + out_features))
        self.weight_ = rng.uniform(-spread, spread, size=(in_features, out_features))
        self.bias_ = rng.uniform(-spread, spread, out_features)

        self._prev_input = None

    def __repr__(self) -> str:
        return f'LinearLayer({self.in_features}, {self.out_features})'

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
        self._prev_input = X

        return X.dot(self.weight_) + self.bias_

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
        # derivative of Cost w.r.t weight
        self._weight_grad = self._prev_input.T.dot(upstream_grad)

        if alpha is not None:
            # Add regularization terms for weights
            self._weight_grad += alpha * self.weight_

        # derivative of Cost w.r.t bias, sum across rows
        self._bias_grad = np.sum(upstream_grad, axis=0)

        # derivative of Cost w.r.t _prev_input
        grad = upstream_grad.dot(self.weight_.T)

        return grad

    def update(self, lr: float) -> None:
        """
        Update the parameters of the layer - only needed for Modular MLPs.

        Parameters
        ----------
        lr : float
            Learning rate for the update
        """
        self.weight_ -= lr * self._weight_grad
        self.bias_ -= lr * self._bias_grad


class ReLULayer:

    def __init__(self):
        self._prev_result = None

    def __call__(self, X: np.ndarray):
        self._prev_result = np.maximum(X, 0)
        return self._prev_result

    def backward(self, upstream_grad: np.ndarray):
        grad = (self._prev_result > 0).astype(float)
        # mask for where the input was positive
        return upstream_grad * grad


class SoftmaxLayer:
    def __init__(self):
        self._prev_result = None

    def __call__(self, X):
        # Subtract the max for numerical stability
        exp_shifted = np.exp(X - np.max(X, axis=1, keepdims=True))
        self._prev_result = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
        return self._prev_result

    def backward(self, upstream_grad):
        grad = self._prev_result * (1 - self._prev_result)
        return upstream_grad * grad


def get_patches(arr: np.ndarray, patch_shape: Tuple[int, int], strides: Tuple[int, int]):
    """Get a strided sub-matrices view of an ndarray.
    Args:
        arr (ndarray): input array of rank 2 or 3, with shape (batch_size, x1, y1, num_channels).
        patch_shape (tuple): window size: (x2, y2).
        strides (int): stride of windows in both y- and x- dimensions.
    Returns:
        subs (view): strided window view of shape (batch_size, *num_patches, x2, y2, num_channels).
    See also skimage.util.shape.view_as_windows()
    """

    assert arr.ndim == 4, 'input array should be rank 4'

    # move channel axis to the front for convenience
    arr = np.moveaxis(arr, -1, 1)
    subs = np.lib.stride_tricks.sliding_window_view(arr, window_shape=patch_shape, axis=(-2, -1))
    subs = np.moveaxis(subs, 1, -1)
    subs = subs[:, ::strides[0], ::strides[1]]

    return subs


def pad_images(X: np.ndarray, padding: int) -> np.ndarray:
    """
    Pad images with zeros on all sides.
    Parameters
    ----------
    X : np.ndarray
        Input images of shape (batch_size, x, y, num_channels)
    padding : int
        Number of pixels to pad on each side

    Returns
    -------
    np.ndarray
        Padded images of shape (batch_size, x + 2 * padding_x, y + 2 * padding_y, num_channels)

    """
    return np.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)))


class ConvLayer:
    """
    Convolutional layer of a convolutional neural network.

    Attributes:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the kernel
        stride: Stride of the convolution
        padding: Padding of the convolution, either "valid" or "same"
        bias_: Bias vector of shape (out_channels,)
        weight_: Weight matrix of shape (out_channels, in_channels, kernel_size, kernel_size)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int],
                 stride: Tuple[int, int] = (1, 1), padding: str = 'same',
                 rng: np.random.RandomState = np.random.RandomState()):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        assert padding in ['valid', 'same'], 'Padding must be either "valid" or "same"'
        self.padding = padding
        spread = (1 / (in_channels * np.prod(kernel_size))) ** .5
        self.bias_ = rng.uniform(-spread, spread, size=(out_channels, 1)).squeeze()
        self.weight_ = rng.uniform(-spread, spread, size=(*kernel_size, in_channels, out_channels))
        self._bias_grad = None
        self._weight_grad = None
        self._prev_input = None

    def _pad(self, X: np.ndarray, padding: Optional[int] = None) -> np.ndarray:
        padding = self.kernel_size[0] // 2 if padding is None else padding
        return np.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)))

    def get_output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Computes the output shape of the layer given an input shape.
        Parameters
        ----------
        input_shape : Sequence[int]
            Shape of the input of the layer (x, y, in_channels)

        Returns
        -------
        Tuple[int, int, int]
            Shape of the output of the layer (x_, y_, out_channels)

        """
        if self.padding == 'same':
            return input_shape[0], input_shape[1], self.out_channels
        x1, y1, _ = input_shape
        x2, y2 = self.kernel_size
        return 1 + (x1 - x2) // self.stride[0], 1 + (y1 - y2) // self.stride[1], self.out_channels

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
        self._prev_input = X
        if self.padding == 'same':
            X = self._pad(X)
        patches = get_patches(X, self.kernel_size, self.stride)

        conv = np.einsum('bnmxyi,xyio->bnmo', patches, self.weight_)

        return conv + self.bias_

    def backward(self, upstream_grad: np.ndarray, alpha: Optional[float] = None) -> np.ndarray:
        """
        Backward pass of the layer
        Parameters
        ----------
        upstream_grad : np.ndarray
            Gradient of the loss w.r.t. the output of the layer of shape (batch_size, width_, height_, out_channels)
        alpha : Optional[float]
            Regularization parameter

        Returns
        -------
        np.ndarray
            Gradient of the loss w.r.t. the input of the layer of shape (batch_size, width, height, in_channels)

        """
        previous_input = self._pad(self._prev_input) if self.padding == 'same' else self._prev_input
        previous_input_patches = get_patches(previous_input, self.kernel_size, self.stride)

        # derivative of Cost w.r.t weight
        self._weight_grad = np.einsum('bnmxyi,bnmo->xyio', previous_input_patches, upstream_grad)

        if alpha is not None:
            # Add regularization terms for weights
            self._weight_grad += alpha * self.weight_

        # derivative of Cost w.r.t bias, sum across pixels
        self._bias_grad = np.sum(upstream_grad, axis=(0, 1, 2))

        # pad upstream grad and get patches for full convolution
        padding = self.kernel_size[0] // 2 if self.padding == 'same' else self.kernel_size[0] - 1
        upstream_grad = self._pad(upstream_grad, padding)
        upstream_grad_patches = get_patches(upstream_grad, self.kernel_size, self.stride)

        weight_flipped = np.flip(self.weight_, axis=(0, 1))

        # derivative of Cost w.r.t _prev_input using full convolution between upstream_grad and flipped kernel
        # REF https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c
        grad = np.einsum('xyio,bnmxyo->bnmi', weight_flipped, upstream_grad_patches)

        assert grad.shape == self._prev_input.shape, f'Need shape {self._prev_input.shape}, got {grad.shape}'
        return grad

    def update(self, lr: float) -> None:
        """
        Update the parameters of the layer

        Parameters
        ----------
        lr : float
            Learning rate for the update
        """
        self.weight_ -= lr * self._weight_grad
        self.bias_ -= lr * self._bias_grad



import numpy as np
from typing import Tuple

def get_patches(arr: np.ndarray, patch_shape: Tuple[int, int], strides: Tuple[int, int]):
    assert arr.ndim == 4, 'input array should be rank 4'
    arr = np.moveaxis(arr, -1, 1)
    subs = np.lib.stride_tricks.sliding_window_view(arr, window_shape=patch_shape, axis=(-2, -1))
    subs = np.moveaxis(subs, 1, -1)
    subs = subs[:, ::strides[0], ::strides[1]]
    return subs

class PoolingLayer:
    def __init__(self, kernel_size: Tuple[int, int], stride: Tuple[int, int] = (2, 2), padding: int = 0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def get_output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        x1, y1, out = input_shape
        x2, y2 = self.kernel_size
        return ((x1 + 2 * self.padding - x2) // self.stride[0] + 1, 
                (y1 + 2 * self.padding - y2) // self.stride[1] + 1, out)

    def __call__(self, X: np.array) -> np.array:
        self._previous_input = X
        if self.padding > 0:
            X = np.pad(X, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        patches = get_patches(X, self.kernel_size, self.stride)
        self._max_mask = (patches == np.max(patches, axis=(3, 4), keepdims=True))
        return np.max(patches, axis=(3, 4))

    def backward(self, upstream_grad: np.ndarray) -> np.ndarray:
        if self.padding > 0:
            self._previous_input = np.pad(self._previous_input, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        patches = get_patches(self._previous_input, self.kernel_size, self.stride)
        grad = np.zeros_like(patches)
        np.putmask(grad, self._max_mask, upstream_grad[..., None, None, :])
        
        grad = grad.reshape(self._previous_input.shape[0], grad.shape[1] * grad.shape[3], grad.shape[2] * grad.shape[4], grad.shape[-1])

        if self.padding > 0:
            grad = grad[:, self.padding:-self.padding, self.padding:-self.padding, :]

        assert grad.shape == self._previous_input.shape, f'Need shape {self._previous_input.shape}, got {grad.shape}'
        return grad

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

    def __init__(self, layers: Sequence[Union[ConvLayer, PoolingLayer]], lr: float = 0.01, epochs: int = 50,
                 random_state: Optional[int] = None, alpha: float = 0.0001, batch_size: int = 32,
                 input_shape: Tuple[int, int, int] = (28, 28, 1)):
        self.layers = layers
        self.lr = lr
        self.epochs = epochs
        self.random_state = random_state
        self.rng_ = np.random.RandomState(random_state)
        self.out_layer_: Optional[ModularLinearLayer] = None
        self.softmax_layer_: Optional[SoftmaxLayer] = None
        self.alpha = alpha
        self.batch_size = batch_size
        self.input_shape = input_shape

    def _forward(self, X: np.array):
        for layer in self.layers:
            X = layer(X)
        X = np.reshape(X, (X.shape[0], -1))
        return self.softmax_layer_(self.out_layer_(X))

    def _calculate_loss(self, logits: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        epsilon = 1e-12
        logits = np.clip(logits, epsilon, 1. - epsilon)
        loss = -np.sum(y * np.log(logits)) / len(y)

        # Add regularization term to loss (L2 / ridge / weight decay)
        loss += self.alpha / 2 * np.sum(
            [np.square(layer.weight_).sum() if hasattr(layer, 'weight_') else 0 for layer in self.layers])

        loss_grad: np.ndarray = -1 * (y - logits) / y.shape[0]

        return loss, loss_grad

    def fit(self, X: np.array, y: np.array):
        num_classes = len(np.unique(y))

        # one-hot encoding for labels
        y = np.eye(num_classes, dtype=int)[y]

        feat_shape = self.input_shape
        for layer in self.layers:
            feat_shape = layer.get_output_shape(feat_shape) if hasattr(layer, 'get_output_shape') else feat_shape

        self.out_layer_ = ModularLinearLayer(int(np.prod(feat_shape)), num_classes, self.rng_)
        self.softmax_layer_ = SoftmaxLayer()

        # Gradient descent. For each batch...
        for epoch in range(0, self.epochs):
            indices = self.rng_.permutation(range(len(X)))
            batches = np.arange(start=0, stop=len(X), step=self.batch_size)[1:]
            for batch_idx in np.array_split(indices, batches):
                X_batch, y_batch = X[batch_idx], y[batch_idx]

                logits = self._forward(X_batch)

                # Calculate loss and its gradient
                batch_loss, grad = self._calculate_loss(logits, y_batch)

                grad = self.softmax_layer_.backward(grad)
                grad = self.out_layer_.backward(grad)
                grad = np.reshape(grad, [-1, *feat_shape])

                # Backpropagation of gradients
                for layer in self.layers[::-1]:
                    grad = layer.backward(grad)

                # Update weights and biases in each layer according to their gradients
                for layer in self.layers:
                    if hasattr(layer, 'update'):
                        layer.update(self.lr)

                self.out_layer_.update(self.lr)

            if epoch % 1 == 0:
                loss, _ = self._calculate_loss(self._forward(X), y)
                print("Epoch {} Loss {}".format(epoch, round(loss, 4)))

        return self

    def predict(self, X: np.array) -> np.array:
        return self._forward(X).argmax(axis=1)


if __name__ == '__main__':
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    rng = np.random.RandomState(42)
    layers = [ConvLayer(1, 7, (3, 3), rng=rng),
              ReLULayer(),
              PoolingLayer((2, 2), stride=(2, 2)),
              ConvLayer(7, 7, (3, 3), rng=rng),
              ]
    cnn = CNNClassifier(layers, random_state=42, input_shape=(8, 8, 1))

    data = load_digits()
    X, y = np.expand_dims(data['images'], axis=-1), data['target']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42, shuffle=True)
    cnn.fit(X_train, y_train)

    print('accuracy', cnn.score(X_test, y_test))