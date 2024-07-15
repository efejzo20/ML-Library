import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """
        Compute the mean and standard deviation to be used for later scaling.
        
        Parameters:
        X : array-like, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation.
        """
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
    
    def transform(self, X):
        """
        Perform standardization by centering and scaling.
        
        Parameters:
        X : array-like, shape [n_samples, n_features]
            The data to be transformed.
        
        Returns:
        X_transformed : array-like, shape [n_samples, n_features]
            The transformed data.
        """
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("This StandardScaler instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
        
        return (X - self.mean_) / self.scale_
    
    def fit_transform(self, X):
        """
        Fit to data, then transform it.
        
        Parameters:
        X : array-like, shape [n_samples, n_features]
            The data to fit and transform.
        
        Returns:
        X_transformed : array-like, shape [n_samples, n_features]
            The transformed data.
        """
        self.fit(X)
        return self.transform(X)
