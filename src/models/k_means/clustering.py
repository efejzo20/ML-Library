from typing import Optional

import numpy as np

from base import BaseEstimator


class KMeans(BaseEstimator):
    def __init__(self, n_clusters: int = 2, max_iter: int = 100, tol: float = 1e-4, random_state: Optional[int] = None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.rng_ = np.random.RandomState(random_state)
        self.centroids_ = None