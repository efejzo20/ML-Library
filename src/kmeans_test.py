from models.k_means.clustering import KMeans
import numpy as np
from sklearn.cluster import KMeans as SklearnKMeans

# Example dataset
X = np.array([
    [1, 2],
    [1, 4],
    [1, 0],
    [4, 2],
    [4, 4],
    [4, 0]
])

# Initialize KMeans with desired parameters
kmeans = KMeans(n_clusters=2, max_iter=100, tol=1e-4, random_state=42)

# Fit the model to the dataset
centroids = kmeans.fit(X)

# Print the resulting centroids
print("Centroids:")
print(centroids)

# sklearn KMeans implementation
sklearn_kmeans = SklearnKMeans(n_clusters=2, max_iter=100, tol=1e-4, random_state=42)
sklearn_kmeans.fit(X)
sklearn_centroids = sklearn_kmeans.cluster_centers_
print("sklearn KMeans Centroids:")
print(sklearn_centroids)