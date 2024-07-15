from typing import Optional
import numpy as np
from abc import ABC, abstractmethod
from base import BaseClassifier
from collections import Counter

class DecisionTree:
    """
    A simple implementation of a Decision Tree classifier.
    
    Parameters:
    -----------
    criterion : str, optional (default='entropy')
        The function to measure the quality of a split. Supported criteria are 
        'gini', 'entropy', and 'misclass'.
    max_depth : int, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until all 
        leaves are pure or until all leaves contain less than min_samples_split samples.
    min_samples_split : int, optional (default=2)
        The minimum number of samples required to split an internal node.
    random_state : int, optional (default=None)
        Controls the randomness of the estimator.
    """
    def __init__(self, criterion='entropy', max_depth=None, min_samples_split=2, random_state=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.rng_ = np.random.RandomState(random_state)
        self.root = None
        self.impurity = self._get_impurity_function()

    def _get_impurity_function(self):
        """
        Selects the impurity function based on the criterion.

        Returns:
        --------
        function
            The impurity function.
        """
        if self.criterion == 'gini':
            return self._gini
        elif self.criterion == 'entropy':
            return self._entropy
        else:
            return self._misclass

    def _gini(self, p):
        """
        Calculates the Gini impurity.

        Parameters:
        -----------
        p : ndarray
            The probabilities of the classes.

        Returns:
        --------
        float
            The Gini impurity.
        """
        return 1. - np.sum(p ** 2)

    def _entropy(self, p):
        """
        Calculates the entropy.

        Parameters:
        -----------
        p : ndarray
            The probabilities of the classes.

        Returns:
        --------
        float
            The entropy.
        """
        idx = np.where(p == 0.)
        p[idx] = 1.
        r = p * np.log2(p)
        return -np.sum(r)
    
    def _misclass(self, p):
        """
        Calculates the misclassification rate.

        Parameters:
        -----------
        p : ndarray
            The probabilities of the classes.

        Returns:
        --------
        float
            The misclassification rate.
        """
        return 1 - np.max(p)
    
    def _information_gain(self, parent, left_child, right_child):
        """
        Calculates the information gain from a split.

        Parameters:
        -----------
        parent : ndarray
            The parent node.
        left_child : ndarray
            The left child node.
        right_child : ndarray
            The right child node.

        Returns:
        --------
        float
            The information gain.
        """
        num_left = len(left_child) / len(parent)
        num_right = len(right_child) / len(parent)
        return self._entropy(parent) - (num_left * self._entropy(left_child) + num_right * self._entropy(right_child))
    
    def _best_split(self, X, y):
        """
        Finds the best split for a node.

        Parameters:
        -----------
        X : ndarray
            The input data.
        y : ndarray
            The target values.

        Returns:
        --------
        dict
            The best split information.
        """
        best_split = {}
        best_info_gain = -1
        n_rows, n_cols = X.shape
        
        for f_idx in range(n_cols):
            X_curr = X[:, f_idx]
            for threshold in np.unique(X_curr):
                df = np.concatenate((X, y.reshape(1, -1).T), axis=1)
                df_left = np.array([row for row in df if row[f_idx] <= threshold])
                df_right = np.array([row for row in df if row[f_idx] > threshold])

                if len(df_left) > 0 and len(df_right) > 0:
                    y_left = df_left[:, -1]
                    y_right = df_right[:, -1]
                    gain = self._information_gain(y, y_left, y_right)
                    if gain > best_info_gain:
                        best_split = {
                            'feature_index': f_idx,
                            'threshold': threshold,
                            'df_left': df_left,
                            'df_right': df_right,
                            'gain': gain
                        }
                        best_info_gain = gain
        return best_split
    
    def _build(self, X, y, depth=0):
        """
        Recursively builds the decision tree.

        Parameters:
        -----------
        X : ndarray
            The input data.
        y : ndarray
            The target values.
        depth : int
            The current depth of the tree.

        Returns:
        --------
        Node
            The root node of the decision tree.
        """
        n_rows, n_cols = X.shape
        
        if n_rows >= self.min_samples_split and (self.max_depth is None or depth < self.max_depth):
            best = self._best_split(X, y)
            if best['gain'] > 0:
                left = self._build(
                    X=best['df_left'][:, :-1], 
                    y=best['df_left'][:, -1], 
                    depth=depth + 1
                )
                right = self._build(
                    X=best['df_right'][:, :-1], 
                    y=best['df_right'][:, -1], 
                    depth=depth + 1
                )
                return Node(
                    feature=best['feature_index'], 
                    threshold=best['threshold'], 
                    data_left=left, 
                    data_right=right, 
                    gain=best['gain']
                )
        return Node(
            value=Counter(y).most_common(1)[0][0]
        )
    
    def fit(self, X, y):
        """
        Fits the decision tree classifier.

        Parameters:
        -----------
        X : ndarray
            The input data.
        y : ndarray
            The target values.
        """
        self.root = self._build(X, y)
        
    def _predict(self, x, tree):
        """
        Predicts a single data point.

        Parameters:
        -----------
        x : ndarray
            A single input data point.
        tree : Node
            The root node of the decision tree.

        Returns:
        --------
        int
            The predicted class.
        """
        if tree.value is not None:
            return tree.value
        feature_value = x[tree.feature]
        if feature_value <= tree.threshold:
            return self._predict(x=x, tree=tree.data_left)
        else:
            return self._predict(x=x, tree=tree.data_right)
        
    def predict(self, X):
        """
        Predicts class labels for the input data.

        Parameters:
        -----------
        X : ndarray
            The input data.

        Returns:
        --------
        list
            The predicted class labels.
        """
        return [self._predict(x, self.root) for x in X]

    def accuracy(self, X, y):
        """
        Calculates the accuracy of the classifier.

        Parameters:
        -----------
        X : ndarray
            The input data.
        y : ndarray
            The target values.

        Returns:
        --------
        float
            The accuracy of the classifier.
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def cross_validate(self, X, y, k=5):
        """
        Performs k-fold cross-validation.

        Parameters:
        -----------
        X : ndarray
            The input data.
        y : ndarray
            The target values.
        k : int, optional (default=5)
            The number of folds.

        Returns:
        --------
        tuple
            The mean and standard deviation of the accuracy.
        """
        folds = self.getKFolds(len(y), k)
        accuracies = []
        for train_idx, test_idx in folds:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            self.fit(X_train, y_train)
            accuracies.append(self.accuracy(X_test, y_test))
        return np.mean(accuracies), np.std(accuracies)

    def getKFolds(self, N, k):
        """
        Generates k folds for cross-validation.

        Parameters:
        -----------
        N : int
            The number of samples.
        k : int
            The number of folds.

        Returns:
        --------
        list
            The indices for the training and test sets for each fold.
        """
        fold_size = N // k
        idx = np.random.permutation(np.arange(fold_size * k))
        splits = np.split(idx, k)
        folds = []
        for i in range(k):
            te = splits[i]
            tr_si = np.setdiff1d(np.arange(k), i)
            tr = np.concatenate([splits[si] for si in tr_si])
            folds.append((tr.astype(np.int), te.astype(np.int)))
        return folds

    def get_params(self, deep=True):
        """
        Gets the parameters of the estimator.

        Parameters:
        -----------
        deep : bool, optional (default=True)
            If True, will return the parameters for this estimator and contained subobjects that are estimators.

        Returns:
        --------
        dict
            The parameters of the estimator.
        """
        return {
            'criterion': self.criterion,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'random_state': self.random_state
        }

    def set_params(self, **params):
        """
        Sets the parameters of the estimator.

        Parameters:
        -----------
        **params : dict
            Estimator parameters.

        Returns:
        --------
        self : object
            Returns self.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

class Node:
    def __init__(self, feature=None, threshold=None, data_left=None, data_right=None, gain=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.data_left = data_left
        self.data_right = data_right
        self.gain = gain
        self.value = value