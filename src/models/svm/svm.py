"""
Module for SVM classifier.

"""
from base import BaseClassifier
import numpy as np
from random import shuffle
from abc import ABC, abstractmethod


class SVM:
    """Implementation of SVM with SGD for regression and classification."""

    def __init__(self, lmbd, D, epsilon=0.1):
        """
        Initialize the SVM model.

        Parameters:
        lmbd (float): Regularization parameter.
        D (int): Number of features.
        epsilon (float): Epsilon parameter for the epsilon-insensitive loss in regression.
        """
        self.lmbd = lmbd  # Regularization parameter
        self.D = D + 1  # Adding one for the bias term
        self.w = np.zeros(self.D)  # Initialize weights to zeros
        self.epsilon = epsilon  # Epsilon parameter for regression

    def epsilon_insensitive_loss(self, target, y):
        """
        Calculate the epsilon-insensitive loss.

        Parameters:
        target (float): Target value.
        y (float): Predicted value.

        Returns:
        float: Epsilon-insensitive loss.
        """
        return max(0, abs(target - y) - self.epsilon)

    def train(self, x, y, alpha):
        """
        Train the SVM model using a single training example.

        Parameters:
        x (array-like): Feature vector.
        y (float or int): Target value or class label.
        alpha (float): Learning rate.

        Returns:
        array-like: Updated weight vector.
        """
        prediction = self.predict(x)  # Make a prediction

        if isinstance(y, int):  # Classification
            if y * prediction < 1:
                # Update weights for misclassified example
                self.w += alpha * (y * x - 2 * self.lmbd * self.w)
            else:
                # Update weights for correctly classified example
                self.w += alpha * (-2 * self.lmbd * self.w)
        else:  # Regression
            if self.epsilon_insensitive_loss(prediction, y) > 0:
                # Update weights for examples outside the epsilon margin
                self.w += alpha * ((y - prediction) * x - 2 * self.lmbd * self.w)
            else:
                # Update weights for examples within the epsilon margin
                self.w += alpha * (-2 * self.lmbd * self.w)
                
        return self.w

    def predict(self, x):
        """
        Make a prediction using the SVM model.

        Parameters:
        x (array-like): Feature vector.

        Returns:
        float: Predicted value.
        """
        return np.dot(self.w, x)

    def sign(self, prediction):
        """
        Determine the sign of the prediction.

        Parameters:
        prediction (float): Predicted value.

        Returns:
        int: 1 if prediction is non-negative, -1 otherwise.
        """
        return 1 if prediction >= 0 else -1


class SVR:
    """Support Vector Regressor."""

    def __init__(self, lr: float = 1.0, tol: float = 1e-4, max_iter: int = 1000, epsilon: float = 0.1):
        """
        Initialize the SVR model.

        Parameters:
        lr (float): Learning rate.
        tol (float): Tolerance for stopping criterion.
        max_iter (int): Maximum number of iterations.
        epsilon (float): Epsilon parameter for the epsilon-insensitive loss.
        """
        self.lr = lr  # Learning rate
        self.tol = tol  # Tolerance for stopping criterion
        self.max_iter = max_iter  # Maximum number of iterations
        self.epsilon = epsilon  # Epsilon parameter for regression
        self.svm = None  # Placeholder for the SVM model

    def fit(self, X, y):
        """
        Fit the SVR model to the training data.

        Parameters:
        X (array-like): Training feature vectors.
        y (array-like): Training target values.
        """
        n_samples, n_features = X.shape
        self.svm = SVM(lmbd=self.lr, D=n_features, epsilon=self.epsilon)  # Initialize the SVM model
        
        for iteration in range(self.max_iter):
            for i in range(n_samples):
                # Calculate learning rate for the current iteration
                alpha = 1.0 / (self.lr * (iteration * n_samples + i + 1))
                self.svm.train(np.append(X[i], 1), y[i], alpha)  # Train the model with the current example

    def predict(self, X):
        """
        Predict target values for the given feature vectors.

        Parameters:
        X (array-like): Feature vectors.

        Returns:
        array-like: Predicted target values.
        """
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)  # Initialize predictions array
        
        for i in range(n_samples):
            predictions[i] = self.svm.predict(np.append(X[i], 1))  # Make a prediction for each example
        
        return predictions

    def evaluate(self, X_test, y_test):
        """
        Evaluate the SVR model on the test data.

        Parameters:
        X_test (array-like): Test feature vectors.
        y_test (array-like): Test target values.

        Returns:
        float: Mean squared error of the predictions.
        """
        predictions = self.predict(X_test)  # Make predictions on the test data
        mse = np.mean((predictions - y_test) ** 2)  # Calculate mean squared error
        return mse


class SVC:
    """Support Vector Classifier."""

    def __init__(self, lr: float = 1.0, tol: float = 1e-4, max_iter: int = 1000):
        """
        Initialize the SVC model.

        Parameters:
        lr (float): Learning rate.
        tol (float): Tolerance for stopping criterion.
        max_iter (int): Maximum number of iterations.
        """
        self.lr = lr  # Learning rate
        self.tol = tol  # Tolerance for stopping criterion
        self.max_iter = max_iter  # Maximum number of iterations
        self.svm = None  # Placeholder for the SVM model

    def fit(self, X, y):
        """
        Fit the SVC model to the training data.

        Parameters:
        X (array-like): Training feature vectors.
        y (array-like): Training class labels.
        """
        n_samples, n_features = X.shape
        self.svm = SVM(lmbd=self.lr, D=n_features)  # Initialize the SVM model
        
        for iteration in range(self.max_iter):
            for i in range(n_samples):
                # Calculate learning rate for the current iteration
                alpha = 1.0 / (self.lr * (iteration * n_samples + i + 1))
                self.svm.train(np.append(X[i], 1), y[i], alpha)  # Train the model with the current example

    def predict(self, X):
        """
        Predict class labels for the given feature vectors.

        Parameters:
        X (array-like): Feature vectors.

        Returns:
        array-like: Predicted class labels.
        """
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)  # Initialize predictions array
        
        for i in range(n_samples):
            predictions[i] = self.svm.sign(self.svm.predict(np.append(X[i], 1)))  # Make a prediction for each example
        
        return predictions

    def evaluate(self, X_test, y_test):
        """
        Evaluate the SVC model on the test data.

        Parameters:
        X_test (array-like): Test feature vectors.
        y_test (array-like): Test class labels.

        Returns:
        tuple: Accuracy, true positive rate, and true negative rate.
        """
        predictions = self.predict(X_test)  # Make predictions on the test data
        accuracy = np.mean(predictions == y_test)  # Calculate accuracy
        tp = np.sum((predictions == 1) & (y_test == 1))  # Calculate true positives
        tn = np.sum((predictions == -1) & (y_test == -1))  # Calculate true negatives
        total_positive = np.sum(y_test == 1)  # Total positive examples
        total_negative = np.sum(y_test == -1)  # Total negative examples
        return accuracy, tp / total_positive, tn / total_negative