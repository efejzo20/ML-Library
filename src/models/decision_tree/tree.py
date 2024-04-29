from typing import Optional
import numpy as np

from base import BaseClassifier


class DecisionTree(BaseClassifier):
    """
    Class for decision tree using the ID3 algorithm.
    """

    def __init__(self, criterion: str = 'entropy', random_state: Optional[int] = None):
        self.criterion = criterion
        self.rng_ = np.random.RandomState(random_state)
