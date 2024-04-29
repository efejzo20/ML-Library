# ML Lab



## A Python Machine Learning Library

A Python library that provides implementations of various machine learning algorithms for tasks such as regression, classification, clustering, and more. 

## Supported Algorithms

1. Linear Regression
2. Logistic Regression
3. k-Means
4. Naive Bayes
5. Decision Trees
6. Ensembles (Random Forest, Gradient Boosting)
7. Support Vector Machines (SVM)
8. Artificial Neural Networks (Multi-layer Perceptron)
9. Advanced Deep Learning (Custom architectures)
10. Convolutional Neural Networks (CNNs)

## Usage

Here's an example of how to use ML-Lib to train a linear regression model:

```python
from linear_regression.regression_sgd import SGDRegression
import numpy as np

# Create some sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Instantiate and train the linear regression model
model = SGDRegression()
model.fit(X, y)

# Make predictions
X_test = np.array([[6], [7], [8]])
predictions = model.predict(X_test)
print(predictions)
```

## License

This project is licensed under the GNU AFFERO GENERAL PUBLIC LICENSE - see the [License](https://git.fim.uni-passau.de/padas/24ss-mllab/fejzo/ml-lab/-/blob/main/LICENSE?ref_type=heads) file for details.
