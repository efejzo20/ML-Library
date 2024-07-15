import numpy as np
from collections import defaultdict
import math

from abc import ABC, abstractmethod
from base import BaseClassifier


class MultinomialNaiveBayes(BaseClassifier):
    """
     Multinomial Naive Bayes

     Parameters
     -----------
     :param x               : np array
                            input training points, size N:D
     :param y               : np array
                            target/expected values
     :param data_info       : dict
                            0: data label/ y (exptected value)
                            doc_count : number of document that belongs to label 0
                            tokens_count : number of tokens the belongs to label 0
                            tokens : all the tokens and their number of occurences in the label 0

                                Eg:. { 0: {'doc_count' : 480,
                                        'tokens_count' : 14552
                                        'tokens' : {'date' : 33, 'from' : 23}
                                        }
                                    }
     :param total_document_count: int
                            Total number of documents in the whole datasets
     :param vocabulary      : dict
                            dictionary of all the vocab present in the datasets including their number of occurences

                                Eg:. {'date' : 20, 'from' : 23}

     :param priors          : dict
                            {'0' : 0.235, '1' : 0.568}
                            0 : category/lass
                            0.235 : prior probability value

                            Prior probability for each class/category/label

                                Eg:. log(p(c=0)) = log(number of items in c = 0) - log(total items in whole datasets)
                                log(p(c=11)) = log(480) - log(11314)

     :param conditionals    : dict
                            {0 : {'date': 0.356,
                                'from' : 0.557}
                            }
                            Conditional probability of each term in input datasets

                                Eg:. conditional probability of a term = log(term count in particular class) - log(token size in a class + size of vocabulary)
                                p(A/B) = p(A intersection B) / p(B)

     :param alpha           : float, Laplace smoothing parameter (default=1.0)
     """

    def __init__(self, alpha: float = 1.0):
        """
        Initialize the classifier with the specified smoothing parameter.

        Parameters:
        alpha : float
            Smoothing parameter to avoid zero probabilities.
        """
        self.x = None  # Training features
        self.y = None  # Training labels
        self.data_info = None  # Information about the training data
        self.total_document_count = None  # Total number of documents
        self.vocabulary = {}  # Vocabulary of all tokens
        self.priors = {}  # Prior probabilities of classes
        self.conditionals = {}  # Conditional probabilities of tokens given classes
        self.alpha = alpha  # Smoothing parameter

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Fit the Multinomial Naive Bayes model according to the given training data.

        Parameters:
        x : np.ndarray
            Training features (documents).
        y : np.ndarray
            Training labels (classes).
        """
        self.x = x
        self.y = y
        self.total_document_count = len(y)  # Total number of documents
        self.data_info = self._compute_data_info(x, y)  # Compute data information
        self.vocabulary = self._build_vocabulary(x)  # Build the vocabulary
        self.priors = self._compute_priors()  # Compute the priors
        self.conditionals = self._compute_conditionals()  # Compute the conditionals

    def _compute_data_info(self, x, y):
        """
        Compute information about the training data.

        Parameters:
        x : np.ndarray
            Training features (documents).
        y : np.ndarray
            Training labels (classes).

        Returns:
        defaultdict
            Dictionary containing document count and token information for each class.
        """
        data_info = defaultdict(lambda: {'doc_count': 0, 'tokens_count': 0, 'tokens': defaultdict(int)})
        for doc, label in zip(x, y):
            data_info[label]['doc_count'] += 1
            for token in doc:
                data_info[label]['tokens'][token] += 1
                data_info[label]['tokens_count'] += 1
        return data_info

    def _build_vocabulary(self, x):
        """
        Build the vocabulary from the training features.

        Parameters:
        x : np.ndarray
            Training features (documents).

        Returns:
        defaultdict
            Dictionary with token counts.
        """
        vocabulary = defaultdict(int)
        for doc in x:
            for token in doc:
                vocabulary[token] += 1
        return vocabulary

    def _compute_priors(self):
        """
        Compute the prior probabilities of classes.

        Returns:
        dict
            Dictionary with prior probabilities for each class.
        """
        priors = {}
        for label, info in self.data_info.items():
            priors[label] = math.log(info['doc_count']) - math.log(self.total_document_count)
        return priors

    def _compute_conditionals(self):
        """
        Compute the conditional probabilities of tokens given classes.

        Returns:
        defaultdict
            Dictionary with conditional probabilities for each token in each class.
        """
        conditionals = defaultdict(lambda: defaultdict(float))
        vocab_size = len(self.vocabulary)
        for label, info in self.data_info.items():
            tokens_count = info['tokens_count']
            for token, count in info['tokens'].items():
                conditionals[label][token] = math.log(count + self.alpha) - math.log(tokens_count + self.alpha * vocab_size)
        return conditionals

    def predict(self, x):
        """
        Perform classification on an array of test vectors X.

        Parameters:
        x : np.ndarray
            Test features (documents).

        Returns:
        np.ndarray
            Predicted class labels for each document.
        """
        predictions = []
        for doc in x:
            label_scores = {}
            for label in self.data_info.keys():
                label_scores[label] = self.priors[label]
                for token in doc:
                    if token in self.vocabulary:
                        label_scores[label] += self.conditionals[label].get(
                            token, 
                            math.log(self.alpha) - math.log(self.data_info[label]['tokens_count'] + self.alpha * len(self.vocabulary))
                        )
            predictions.append(max(label_scores, key=label_scores.get))
        return np.array(predictions)

    def predict_proba(self, x):
        """
        Return probability estimates for the test vector X.

        Parameters:
        x : np.ndarray
            Test features (documents).

        Returns:
        list of dicts
            Probability estimates for each document.
        """
        probabilities = []
        for doc in x:
            label_scores = {}
            for label in self.data_info.keys():
                label_scores[label] = self.priors[label]
                for token in doc:
                    if token in self.vocabulary:
                        label_scores[label] += self.conditionals[label].get(
                            token, 
                            math.log(self.alpha) - math.log(self.data_info[label]['tokens_count'] + self.alpha * len(self.vocabulary))
                        )
            max_score = max(label_scores.values())
            exp_scores = {label: math.exp(score - max_score) for label, score in label_scores.items()}
            total_exp_scores = sum(exp_scores.values())
            probabilities.append({label: score / total_exp_scores for label, score in exp_scores.items()})
        return probabilities

