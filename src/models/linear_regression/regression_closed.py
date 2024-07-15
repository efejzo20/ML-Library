import numpy as np
from numpy import linalg as la
from abc import ABC, abstractmethod
from base import BaseRegressor

class ClosedFormRegression(BaseRegressor):
    def __init__(self, bias=True):
        """
        X: data matrix n rows, d columns
        y: target values n rows, 1 column
        bias: if true, a bias term is included at the end of the weight vector, default is True
        """
        
        if not isinstance(bias, bool):
            raise ValueError("bias must be a boolean value.")
        
        self.bias = bias
        self.weights = None
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None

    def fit(self, X, y):
        """
        Fit the regression model to the training data.
        
        Parameters:
        - X (numpy.ndarray): Data matrix of shape (n_samples, n_features).
        - y (numpy.ndarray): Target values of shape (n_samples,).
        """
        X, y = self._validate_input(X, y)

        # Standardize X and y
        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0)
        self.y_mean = y.mean(axis=0)
        self.y_std = y.std(axis=0)

        X_standardized = (X - self.X_mean) / self.X_std
        y_standardized = (y - self.y_mean) / self.y_std

        # Add bias term if specified
        if self.bias:
            X_standardized = np.concatenate((X_standardized, np.ones((X_standardized.shape[0], 1))), axis=1)

        # Solve for weights using np.linalg.solve for better numerical stability
        A = X_standardized.T.dot(X_standardized)
        b = X_standardized.T.dot(y_standardized)
        self.weights = la.solve(A, b)

    def predict(self, X):
        """
        Predict target values for the given input data.
        
        Parameters:
        - X (numpy.ndarray): Data matrix of shape (n_samples, n_features).
        
        Returns:
        - numpy.ndarray: Predicted target values of shape (n_samples,).
        """
        X_standardized = (X - self.X_mean) / self.X_std

        # Add bias term if specified
        if self.bias:
            X_standardized = np.concatenate((X_standardized, np.ones((X_standardized.shape[0], 1))), axis=1)

        y_standardized_pred = np.dot(X_standardized, self.weights)
        return y_standardized_pred * self.y_std + self.y_mean

    def score(self, X, y):
        """
        Calculate the R^2 score of the model.
        
        The R^2 score, also known as the coefficient of determination, is a measure of how well the regression model
        explains the variability of the target variable. It is calculated as:
        
        R^2 = 1 - (SS_res / SS_tot)
        
        Where:
        - SS_res (Residual Sum of Squares) is the sum of the squared differences between the observed target values and 
          the predicted target values.
        - SS_tot (Total Sum of Squares) is the sum of the squared differences between the observed target values and the 
          mean of the observed target values.
        
        Parameters:
        - X (numpy.ndarray): Data matrix of shape (n_samples, n_features).
        - y (numpy.ndarray): True target values of shape (n_samples,).
        
        Returns:
        - float: R^2 score.
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        return 1 - (ss_res / ss_tot)
    
    @staticmethod
    def pdiff(y, y_pred):
        """
        Calculate the prediction difference.
        
        Parameters:
        - y (numpy.ndarray): True target values.
        - y_pred (numpy.ndarray): Predicted target values.
        
        Returns:
        - numpy.ndarray: Difference between true and predicted target values.
        """
        return y - y_pred
    
    @staticmethod
    def rss(pdiff):
        """
        Calculate the residual sum of squares (RSS).
        
        Parameters:
        - pdiff (numpy.ndarray): Prediction differences.
        
        Returns:
        - float: Residual sum of squares.
        """
        return np.sum(pdiff ** 2)
    
    def _validate_input(self, X, y=None, fitting=True):
        """
        Validate the input arrays X and y.
        
        Parameters:
        - X (numpy.ndarray): Data matrix.
        - y (numpy.ndarray, optional): Target values.
        - fitting (bool): Whether the validation is for fitting or prediction.
        
        Returns:
        - X (numpy.ndarray): Validated data matrix.
        - y (numpy.ndarray, optional): Validated target values if fitting.
        
        Raises:
        - ValueError: If input arrays have incorrect shapes or types.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be a numpy array.")
        
        if fitting:
            if y is None:
                raise ValueError("y cannot be None when fitting.")
            if not isinstance(y, np.ndarray):
                raise ValueError("y should be a numpy array.")
            if X.shape[0] != y.shape[0]:
                raise ValueError("The number of samples in X and y should be the same.")
            return X, y
        else:
            return X