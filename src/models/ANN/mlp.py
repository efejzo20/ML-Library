from typing import Sequence, Optional
from abc import ABC, abstractmethod
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
        self.in_features = in_features
        self.out_features = out_features
        scale = np.sqrt(6.0 / (in_features + out_features))
        self.weight_ = rng.uniform(-scale, scale, size=(in_features, out_features))
        self.bias_ = rng.uniform(-scale, scale, out_features)
        self._prev_input = None
        self._weight_grad = None
        self._bias_grad = None

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
        self._weight_grad = self._prev_input.T.dot(upstream_grad)
        if alpha:
            self._weight_grad += alpha * self.weight_
        self._bias_grad = np.sum(upstream_grad, axis=0)
        return upstream_grad.dot(self.weight_.T)

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

class SoftmaxLayer:
    def __init__(self):
        self._prev_result = None

    def __call__(self, X):
        exps = np.exp(X - np.max(X, axis=1, keepdims=True))
        self._prev_result = exps / np.sum(exps, axis=1, keepdims=True)
        return self._prev_result

    def backward(self, upstream_grad):
        return upstream_grad

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
        return upstream_grad * (1 - np.square(self._prev_result))

class ReLULayer:
    def __init__(self):
        self._prev_result = None

    def __call__(self, X):
        self._prev_result = np.maximum(0, X)
        return self._prev_result

    def backward(self, upstream_grad):
        return upstream_grad * (self._prev_result > 0)

class SigmoidLayer:
    def __init__(self):
        self._prev_result = None

    def __call__(self, X):
        self._prev_result = 1 / (1 + np.exp(-X))
        return self._prev_result

    def backward(self, upstream_grad):
        return upstream_grad * self._prev_result * (1 - self._prev_result)


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
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.alpha = alpha
        self.activation = activation
        self.layers_ = None

    def _get_activation_layer(self):
        if self.activation == 'tanh':
            return TanhLayer()
        elif self.activation == 'relu':
            return ReLULayer()
        elif self.activation == 'sigmoid':
            return SigmoidLayer()
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

    def fit(self, X: np.array, y: np.array):
        in_size = X.shape[1]
        self.layers_ = []
        for size in self.hidden_layer_sizes:
            self.layers_.append(ModularLinearLayer(in_size, size, self.rng))
            in_size = size
            self.layers_.append(self._get_activation_layer())
        self.layers_.append(ModularLinearLayer(in_size, 1, self.rng))

        for epoch in range(self.epochs):
            # Forward pass
            output = X
            for layer in self.layers_:
                output = layer(output)

            # Calculate loss (MSE)
            loss = np.mean((output - y) ** 2) / 2

            # Backward pass
            grad = output - y
            for layer in reversed(self.layers_):
                if isinstance(layer, ModularLinearLayer):
                    grad = layer.backward(grad, self.alpha)
                else:
                    grad = layer.backward(grad)

            # Update weights
            for layer in self.layers_:
                if isinstance(layer, ModularLinearLayer):
                    layer.update(self.lr)

            # Print loss every 100 epochs
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

        return self
    
    def predict(self, X: np.array) -> np.array:
        output = X
        for layer in self.layers_:
            output = layer(output)
        return output
    
class MLPClassifier:
    """
    A simple Multi-Layer Perceptron (MLP) classifier with customizable hidden layers, learning rate, and training epochs.

    Parameters
    ----------
    hidden_layer_sizes : Sequence[int], default=(10, 10)
        The number of units in each hidden layer.
    lr : float, default=0.01
        Learning rate for weight updates.
    epochs : int, default=1000
        Number of training epochs.
    random_state : Optional[int], default=None
        Seed for random number generation for reproducibility.

    Attributes
    ----------
    layers_ : List
        List containing the layers of the MLP.
    activation_ : SoftmaxLayer
        The softmax activation layer for the output.
    rng : np.random.RandomState
        Random number generator initialized with random_state.
    """

    def __init__(self, hidden_layer_sizes: Sequence[int] = (10, 10), lr: float = 0.01, epochs: int = 1000, random_state: Optional[int] = None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.lr = lr
        self.epochs = epochs
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.layers_ = None

    def _get_activation_layer(self):
        """
        Returns an activation layer to be used in the network.
        
        Returns
        -------
        TanhLayer
            An instance of the Tanh activation layer.
        """
        return TanhLayer()

    def fit(self, X: np.array, y: np.array):
        """
        Trains the MLP classifier on the provided data.

        Parameters
        ----------
        X : np.array
            Training data of shape (n_samples, n_features).
        y : np.array
            Target values of shape (n_samples,).

        Returns
        -------
        self : MLPClassifier
            Returns the instance itself.
        """
        num_classes = np.unique(y).shape[0]
        in_size = X.shape[1]
        self.layers_ = []
        
        # Create hidden layers
        for size in self.hidden_layer_sizes:
            self.layers_.append(ModularLinearLayer(in_size, size, self.rng))
            self.layers_.append(self._get_activation_layer())
            in_size = size
        
        # Create output layer
        self.layers_.append(ModularLinearLayer(in_size, num_classes, self.rng))
        self.activation_ = SoftmaxLayer()

        y_onehot = self._one_hot_encode(y, num_classes)

        for epoch in range(self.epochs):
            logits = X
            # Forward pass
            for layer in self.layers_:
                logits = layer(logits)
            probs = self.activation_(logits)
            errors = probs - y_onehot
            grad = errors / X.shape[0]

            # Backward pass
            for layer in reversed(self.layers_):
                if isinstance(layer, ModularLinearLayer):
                    grad = layer.backward(grad)
                else:
                    grad = layer.backward(grad)
            
            # Update weights
            for layer in self.layers_:
                if isinstance(layer, ModularLinearLayer):
                    layer.update(self.lr)

            # Compute cross-entropy loss
            loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-15), axis=1))  # Cross-entropy loss
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

        return self

    def predict(self, X: np.array) -> np.array:
        """
        Predicts the class labels for the provided data.

        Parameters
        ----------
        X : np.array
            Input data of shape (n_samples, n_features).

        Returns
        -------
        np.array
            Predicted class labels of shape (n_samples,).
        """
        logits = X
        for layer in self.layers_:
            logits = layer(logits)
        probs = self.activation_(logits)
        return np.argmax(probs, axis=1)

    @staticmethod
    def _one_hot_encode(y, num_classes):
        """
        Converts class labels to one-hot encoded vectors.

        Parameters
        ----------
        y : np.array
            Class labels of shape (n_samples,).
        num_classes : int
            Number of unique classes.

        Returns
        -------
        np.array
            One-hot encoded matrix of shape (n_samples, num_classes).
        """
        return np.eye(num_classes)[y]


