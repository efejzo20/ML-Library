import numpy as np

from base import BaseClassifier

class LogisticRegression(BaseClassifier):
    """
    Logistic Regression.
    """

    def __init__(self, lr: float = 1e-3, epochs: int = 100):
        self.lr = lr
        self.epochs = epochs
        self.w_ = np.array([])
        self.w_history_ = np.array([])
        self.w_0_ = 0