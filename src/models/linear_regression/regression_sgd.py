import numpy as np
from base import BaseRegressor 
 
class SGDRegression(BaseRegressor):
    def __init__(self, learning_rate=0.1, n_iterations=100, mb=100, debug=False):
        """
        X: data matrix n rows, d columns
        y: target values n rows, 1 column
        learning_rate: learning rate, default is 0.1
        n_iterations: fixed number of iterations as stopping criterion, default is 3000
        mb: number of examples in a minibatch, default is 100
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.mb = mb
        self.debug = debug
        self.weights = None

    def fit(self, X, y):
        X_ = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)  # Add bias feature with value 1
        n, d = X_.shape
        self.weights = np.ones((d, 1))  # Initialize weight vector
        self.weights = np.random.normal(0, 0.5, (d, 1))  # Optional: Initialize weights randomly

        for e in range(self.n_iterations):
            ix = np.random.choice(n, size=self.mb, replace=False)
            yi_hat = np.dot(X_[ix], self.weights)
            error = y[ix] - yi_hat
            mse = np.mean(error ** 2.)
            gw = (1. / self.mb) * 2. * np.dot(X_[ix].T, error)

            if self.debug and e % 10 == 0:
                print('MSE:', mse)
                print('Weights:', self.weights.flatten())
                print()

            self.weights = self.weights + self.learning_rate * gw


    def predict(self, X):
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        return np.dot(X, self.weights)
    
    def standardize(self, X):
        return (X - X.mean(axis=0)) / X.std(axis=0)

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    

    def pdiff(self, y, y_pred):
        return (y - y_pred).flatten()
    
    def rss(self, pdiff):
        return np.sum(pdiff ** 2.)