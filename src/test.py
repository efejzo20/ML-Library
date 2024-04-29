# import liner regression model

from models.linear_regression.regression_closed import ClosedFormRegression
from models.linear_regression.regression_sgd import SGDRegression
import pandas as pd

data = pd.read_csv("data/auto-mpg.csv")
X = data[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year']].values
y = data[['mpg']].values

X_max, y_max = X.max(axis=0), y.max(axis=0)
X_min, y_min = X.min(axis=0), y.min(axis=0)
X_mean, y_mean = X.mean(axis=0), y.mean(axis=0)
X_std, y_std = X.std(axis=0), y.std(axis=0)

# Standardization
Xs = (X - X_mean) / X_std


model1 = SGDRegression()

model1.fit(Xs, y)
y_pred = model1.predict(Xs)

# print weights
print("Weights",model1.weights)

# print y_pred
print("y_pred",y_pred)

pdiff = model1.pdiff(y, y_pred)
print("Pdiff",pdiff)

rss = model1.rss(pdiff)
print("RSS",rss)


# do the same for model2
model2 = ClosedFormRegression()
model2.fit(X, y)
y_pred = model2.predict(X)

# print weights
print("Weights 2:",model2.weights)

# print y_pred
print("y_pred 2:",y_pred)

pdiff = model2.pdiff(y, y_pred)

print("Pdiff 2:",pdiff)

rss = model2.rss(pdiff)
print("RSS 2:",rss)