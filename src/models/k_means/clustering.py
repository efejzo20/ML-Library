from typing import Optional
from abc import ABC, abstractmethod
import numpy as np


from base import BaseEstimator


class KMeans(BaseEstimator):
    """
    K-Means clustering algorithm.

    Parameters:
    n_clusters (int): The number of clusters to form.
    max_iter (int): Maximum number of iterations of the k-means algorithm.
    tol (float): Tolerance to declare convergence.
    random_state (Optional[int]): Seed for random number generator.

    Attributes:
    rng_ (np.random.Generator): Random number generator.
    centroids_ (np.ndarray): Coordinates of cluster centers.
    """
    
    def __init__(self, n_clusters: int = 2, max_iter: int = 100, tol: float = 1e-4, random_state: Optional[int] = None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.rng_ = np.random.default_rng(random_state)  # Random number generator
        self.centroids_ = None  # To store the centroids

    def get_initial_centroids(self, X):
        """
        Initialize centroids by randomly selecting samples from the dataset.

        Parameters:
        X (np.ndarray): Dataset to cluster.

        Returns:
        np.ndarray: Initial centroids.
        """
        num_samples = X.shape[0]
        sample_pt_idx = self.rng_.choice(num_samples, self.n_clusters, replace=False)
        centroids = [tuple(X[id]) for id in sample_pt_idx]
        unique_centroids = list(set(centroids))
        
        # Ensure unique centroids in case of duplicates
        while len(unique_centroids) < self.n_clusters:
            new_sample_pt_idx = self.rng_.choice(num_samples, self.n_clusters - len(unique_centroids), replace=False)
            new_centroids = [tuple(X[id]) for id in new_sample_pt_idx]
            unique_centroids = list(set(unique_centroids + new_centroids))
        
        return np.array(unique_centroids)

    def euclidean_distance(self, A, B):
        """
        Compute the Euclidean distance between two matrices.

        Parameters:
        A (np.ndarray): First matrix.
        B (np.ndarray): Second matrix.

        Returns:
        np.ndarray: Matrix of distances.
        """
        A_square = np.reshape(np.sum(A * A, axis=1), (A.shape[0], 1))
        B_square = np.reshape(np.sum(B * B, axis=1), (1, B.shape[0]))
        AB = A @ B.T
        C = -2 * AB + B_square + A_square
        return np.sqrt(C)

    def compute_clusters(self, X, centroids):
        """
        Assign samples in X to the nearest centroids to form clusters.

        Parameters:
        X (np.ndarray): Dataset to cluster.
        centroids (np.ndarray): Current centroids.

        Returns:
        dict: Clusters with assigned samples.
        """
        k = centroids.shape[0]
        clusters = {i: [] for i in range(k)}
        distance_mat = self.euclidean_distance(X, centroids)
        closest_cluster_ids = np.argmin(distance_mat, axis=1)
        
        for i, cluster_id in enumerate(closest_cluster_ids):
            clusters[cluster_id].append(X[i])
        
        return clusters

    def check_convergence(self, previous_centroids, new_centroids):
        """
        Check if the algorithm has converged.

        Parameters:
        previous_centroids (np.ndarray): Centroids from the previous iteration.
        new_centroids (np.ndarray): Current centroids.

        Returns:
        bool: True if the algorithm has converged, False otherwise.
        """
        distances_between_old_and_new_centroids = self.euclidean_distance(previous_centroids, new_centroids)
        converged = np.max(distances_between_old_and_new_centroids.diagonal()) <= self.tol
        return converged

    def fit(self, X, y=None):
        """
        Compute k-means clustering.

        Parameters:
        X (np.ndarray): Dataset to cluster.
        y (np.ndarray, optional): Ignored, present here for API consistency by convention.

        Returns:
        np.ndarray: Centroids of the clusters.
        """
        self.centroids_ = self.get_initial_centroids(X)
        converged = False
        iterations = 0
        
        while not converged and iterations < self.max_iter:
            previous_centroids = self.centroids_
            clusters = self.compute_clusters(X, previous_centroids)
            self.centroids_ = np.array([np.mean(clusters[key], axis=0, dtype=X.dtype) for key in sorted(clusters.keys())])
            converged = self.check_convergence(previous_centroids, self.centroids_)
            iterations += 1
        
        return self.centroids_