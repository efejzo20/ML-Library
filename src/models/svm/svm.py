"""
Module for SVM classifier (Session 08).

"""
from base import BaseClassifier


class SVC(BaseClassifier):
    def __init__(self, lr: float = 1.0, tol: float = 1e-4, max_iter: int = 1000, kernel:str = 'linear'):
        self.lr = lr
        self.tol = tol
        self.max_iter = max_iter
        self.kernel = kernel


class SVR(SVC):
    def __init__(self, lr: float = 1.0, tol: float = 1e-4, max_iter: int = 1000, kernel:str = 'linear'):
        super().__init__(lr=lr, tol=tol, max_iter=max_iter, kernel=kernel)
