from typing import Optional
import numpy as np

from base import BaseClassifier
from tree import DecisionTree  # Import the DecisionTree from previous session (Optional)

class RandomForestClassifier(BaseClassifier):
    def __init__(self, n_estimators: int = 100, max_depth: int = None, max_features: str = 'sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.estimators_ = []
        self.classes_ = None


class GradientBoostingClassifier(BaseClassifier):
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, random_state: Optional[int] = None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.rng_ = np.random.RandomState(random_state)
