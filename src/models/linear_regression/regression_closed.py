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
        self.bias = bias
        self.weights = None

    def fit(self, X, y):
        if self.bias:
            X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        self.weights = la.inv(X.T.dot(X)).dot(X.T).dot(y)

    def predict(self, X):
        if self.bias:
            X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        return np.dot(X, self.weights)

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def standardize(self, X):
        return (X - X.mean(axis=0)) / X.std(axis=0)
    
    def pdiff(self, y, y_pred):
        return y - y_pred
    
    def rss(self, pdiff):
        return np.sum(pdiff ** 2)