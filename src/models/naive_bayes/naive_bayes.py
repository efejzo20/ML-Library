import numpy as np

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
        self.x = []
        self.y = []
        self.data_info = None
        self.total_document_count = None
        self.vocabulary = {}
        self.priors = {}
        self.conditionals = {}
        self.alpha = alpha




