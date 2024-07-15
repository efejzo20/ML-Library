import numpy as np
from base import BaseRegressor 
 
class SGDRegression(BaseRegressor):
    def __init__(self, learning_rate=0.1, n_iterations=100, mb=100, debug=False, gradient_clip=1.0, learning_rate_decay=0.99):
        """
        Initialize the SGDRegression model with given parameters.

        Parameters:
        learning_rate (float): Learning rate for gradient descent.
        n_iterations (int): Number of iterations for training.
        mb (int): Number of examples in a minibatch.
        debug (bool): If True, print debug information during training.
        gradient_clip (float): Maximum value for gradient clipping.
        learning_rate_decay (float): Decay factor for learning rate.

        Raises:
        ValueError: If any of the input parameters are out of their expected range.
        """
        if learning_rate <= 0 or learning_rate > 1:
            raise ValueError("learning_rate must be between 0 and 1.")
        if n_iterations <= 0:
            raise ValueError("n_iterations must be greater than 0.")
        if mb <= 0:
            raise ValueError("mb (mini-batch size) must be greater than 0.")
        if gradient_clip <= 0:
            raise ValueError("gradient_clip must be greater than 0.")
        if not (0 < learning_rate_decay <= 1):
            raise ValueError("learning_rate_decay must be between 0 and 1.")

        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.mb = mb
        self.debug = debug
        self.weights = None
        self.gradient_clip = gradient_clip
        self.learning_rate_decay = learning_rate_decay
        self.mean = None
        self.std = None

    def _add_bias_term(self, X):
        """
        Add a bias term (column of ones) to the data matrix X.

        Parameters:
        X (numpy.ndarray): Data matrix (n samples, d features).

        Returns:
        numpy.ndarray: Data matrix with added bias term (n samples, d+1 features).
        """
        return np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

    def _standardize(self, X):
        """
        Standardize the data X using the stored mean and standard deviation.

        Parameters:
        X (numpy.ndarray): Data matrix (n samples, d features).

        Returns:
        numpy.ndarray: Standardized data.
        """
        return (X - self.mean) / self.std

    def fit(self, X, y):
        """
        Fit the model to the data X and target y using stochastic gradient descent.

        Parameters:
        X (numpy.ndarray): Data matrix (n samples, d features).
        y (numpy.ndarray): Target values (n samples, 1).

        Raises:
        ValueError: If the number of samples in X and y are not equal.
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("The number of samples in X and y must be equal.")

        # Standardize the features
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        X_standardized = self._standardize(X)
        
        # Add bias term
        X_ = self._add_bias_term(X_standardized)
        n, d = X_.shape

        # Ensure mini-batch size does not exceed the number of samples
        self.mb = min(self.mb, n)
        
        # Initialize weights randomly
        self.weights = np.random.normal(0, 0.5, (d, 1))

        for e in range(self.n_iterations):
            # Sample a mini-batch
            ix = np.random.choice(n, size=self.mb, replace=False)
            yi_hat = np.dot(X_[ix], self.weights)
            error = y[ix] - yi_hat
            mse = np.mean(error ** 2)
            
            # Compute the gradient
            gw = (2. / self.mb) * np.dot(X_[ix].T, error)
            
            # Gradient clipping
            norm_gw = np.linalg.norm(gw)
            if norm_gw > self.gradient_clip:
                gw = (self.gradient_clip / norm_gw) * gw

            # Debugging information
            if self.debug and e % 10 == 0:
                print(f'Iteration {e}, MSE: {mse}')
                print(f'Weights: {self.weights.flatten()}')
                print(f'Gradient Norm: {norm_gw}')
                print()

            # Update weights
            self.weights += self.learning_rate * gw
            self.learning_rate *= self.learning_rate_decay  # Decay the learning rate

    def predict(self, X):
        """
        Predict target values for given data X.

        Parameters:
        X (numpy.ndarray): Data matrix (n samples, d features).

        Returns:
        numpy.ndarray: Predicted target values (n samples, 1).
        """
        X_standardized = self._standardize(X)
        X_ = self._add_bias_term(X_standardized)
        return np.dot(X_, self.weights)

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

    def pdiff(self, y, y_pred):
        """
        Calculate the prediction difference.

        Parameters:
        y (numpy.ndarray): True target values (n samples, 1).
        y_pred (numpy.ndarray): Predicted target values (n samples, 1).

        Returns:
        numpy.ndarray: Prediction difference (n samples,).
        """
        return (y - y_pred).flatten()

    def rss(self, pdiff):
        """
        Calculate the residual sum of squares.

        Parameters:
        pdiff (numpy.ndarray): Prediction differences (n samples,).

        Returns:
        float: Residual sum of squares.
        """
        return np.sum(pdiff ** 2)