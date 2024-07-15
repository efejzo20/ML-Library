from typing import Optional
import numpy as np
from collections import Counter
from abc import ABC, abstractmethod
from base import BaseClassifier
from models.decision_tree.tree import DecisionTree   

class RandomForest:
    """
    Random Forest Classifier

    Attributes:
        num_trees (int): Number of trees in the forest.
        min_samples_split (int): Minimum number of samples required to split an internal node.
        max_depth (int): Maximum depth of the trees.
        random_state (Optional[int]): Seed for the random number generator.
    """

    def __init__(self, num_trees=25, min_samples_split=2, max_depth=5, random_state=None):
        self.num_trees = num_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.random_state = random_state
        self.decision_trees = []

    def _sample(self, X, y):
        """
        Generate a bootstrap sample from the dataset.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Bootstrap sample of the feature matrix and target vector.
        """
        n_rows, n_cols = X.shape
        samples = np.random.choice(a=n_rows, size=n_rows, replace=True)
        return X[samples], y[samples]

    def fit(self, X, y):
        """
        Fit the random forest model to the training data.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
        """
        if len(self.decision_trees) > 0:
            self.decision_trees = []

        for _ in range(self.num_trees):
            clf = DecisionTree(
                criterion='entropy', 
                max_depth=self.max_depth, 
                min_samples_split=self.min_samples_split,
                random_state=self.random_state
            )
            _X, _y = self._sample(X, y)
            clf.fit(_X, _y)
            self.decision_trees.append(clf)

    def predict(self, X):
        """
        Predict the class labels for the given data.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            List[int]: Predicted class labels.
        """
        y = []
        for tree in self.decision_trees:
            y.append(tree.predict(X))
        
        y = np.swapaxes(a=y, axis1=0, axis2=1)
        predictions = []
        for preds in y:
            counter = Counter(preds)
            predictions.append(counter.most_common(1)[0][0])
        return predictions

    def accuracy(self, X, y):
        """
        Compute the accuracy of the model.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): True class labels.

        Returns:
            float: Accuracy of the model.
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def cross_validate(self, X, y, k=5):
        """
        Perform k-fold cross-validation.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
            k (int): Number of folds.

        Returns:
            Tuple[float, float]: Mean and standard deviation of the accuracies across the folds.
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
        Generate k-fold splits for cross-validation.

        Args:
            N (int): Number of samples.
            k (int): Number of folds.

        Returns:
            List[Tuple[np.ndarray, np.ndarray]]: List of train and test indices for each fold.
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
        Get the parameters of the model.

        Args:
            deep (bool): If True, return the parameters for this estimator and contained subobjects that are estimators.

        Returns:
            dict: Parameters of the model.
        """
        return {
            'num_trees': self.num_trees,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'random_state': self.random_state
        }

    def set_params(self, **params):
        """
        Set the parameters of the model.

        Args:
            **params: Dictionary of model parameters.
            
        Returns:
            self: The model with updated parameters.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

class GradientBoostingClassifier(BaseClassifier):
    """
    Gradient Boosting Classifier

    Attributes:
        n_estimators (int): Number of boosting stages to be run.
        learning_rate (float): Learning rate shrinks the contribution of each tree.
        max_depth (int): Maximum depth of the individual regression estimators.
        random_state (Optional[int]): Seed for the random number generator.
    """

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3, random_state: Optional[int] = None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.rng_ = np.random.RandomState(random_state)
        self.estimators_ = []
        self.classes_ = None

    def fit(self, X, y):
        """
        Fit the gradient boosting model to the training data.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
        """
        self.classes_ = np.unique(y)
        n_samples, n_classes = X.shape[0], len(self.classes_)
        F = np.zeros((n_samples, n_classes))

        for _ in range(self.n_estimators):
            residual = self._gradient(F, y)
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X, residual)
            self.estimators_.append(tree)
            F += self.learning_rate * tree.predict(X).reshape(-1, 1)

    def predict(self, X):
        """
        Predict the class labels for the given data.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted class labels.
        """
        F = np.zeros((X.shape[0], len(self.classes_)))
        for tree in self.estimators_:
            F += self.learning_rate * tree.predict(X).reshape(-1, 1)
        return self.classes_.take(np.argmax(F, axis=1), axis=0)

    def _gradient(self, F, y):
        """
        Compute the gradient of the loss function.

        Args:
            F (np.ndarray): Predicted values.
            y (np.ndarray): True class labels.

        Returns:
            np.ndarray: Gradient of the loss function.
        """
        one_hot_y = np.eye(len(self.classes_))[y]
        return one_hot_y - self._softmax(F)

    def _softmax(self, F):
        """
        Compute the softmax function.

        Args:
            F (np.ndarray): Predicted values.

        Returns:
            np.ndarray: Softmax probabilities.
        """
        exp_F = np.exp(F - np.max(F, axis=1, keepdims=True))
        return exp_F / np.sum(exp_F, axis=1, keepdims=True)