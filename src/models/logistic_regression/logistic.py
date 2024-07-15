import numpy as np
from base import BaseClassifier
from abc import ABC, abstractmethod

class LogisticRegression(BaseClassifier):
    """
    Logistic Regression classifier.

    Parameters
    ----------
    learning_rate : float, optional (default=0.01)
        The learning rate for gradient descent optimization.
    num_iterations : int, optional (default=1000)
        The number of iterations for the optimization loop.

    Attributes
    ----------
    weights : ndarray of shape (n_features,)
        The weights for the logistic regression model.
    bias : float
        The bias term for the logistic regression model.
    """

    def __init__(self, learning_rate=0.01, num_iterations=1000):
        """
        Initialize the LogisticRegression classifier with given learning rate and number of iterations.

        Parameters
        ----------
        learning_rate : float, optional
            The learning rate for gradient descent optimization.
        num_iterations : int, optional
            The number of iterations for the optimization loop.
        """

        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be between 0 and 1.")
        if not isinstance(num_iterations, int) or num_iterations <= 0:
            raise ValueError("num_iterations must be a positive integer.")
        
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        """
        Compute the sigmoid function.

        Parameters
        ----------
        z : ndarray or scalar
            Input value or array of values.

        Returns
        -------
        ndarray or scalar
            Sigmoid function output.
        """
        return 1 / (1 + np.exp(-z))
    
    def initialize_parameters(self, n_features):
        """
        Initialize the weights and bias parameters.

        Parameters
        ----------
        n_features : int
            Number of features in the input data.
        """
        self.weights = np.zeros(n_features)
        self.bias = 0
    
    def compute_gradient(self, X, y):
        """
        Compute the gradient of the loss function with respect to weights and bias.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        dw : ndarray of shape (n_features,)
            Gradient of the loss with respect to weights.
        db : float
            Gradient of the loss with respect to bias.
        """
        m = X.shape[0]
        y_pred = self.sigmoid(np.dot(X, self.weights) + self.bias)
        dw = (1 / m) * np.dot(X.T, (y_pred - y))
        db = (1 / m) * np.sum(y_pred - y)
        return dw, db
    
    def update_parameters(self, dw, db):
        """
        Update the weights and bias parameters using gradient descent.

        Parameters
        ----------
        dw : ndarray of shape (n_features,)
            Gradient of the loss with respect to weights.
        db : float
            Gradient of the loss with respect to bias.
        """
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
    
    def fit(self, X, y):
        """
        Fit the logistic regression model to the training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.
        """
        self.initialize_parameters(X.shape[1])
        
        for i in range(self.num_iterations):
            dw, db = self.compute_gradient(X, y)
            self.update_parameters(dw, db)
    
    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels.
        """
        y_pred = self.sigmoid(np.dot(X, self.weights) + self.bias)
        return np.round(y_pred)
