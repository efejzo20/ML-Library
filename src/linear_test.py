# import liner regression model

from models.linear_regression.regression_closed import ClosedFormRegression
from models.linear_regression.regression_sgd import SGDRegression
from StandardScaler.StandardScaler import StandardScaler
import pandas as pd

data = pd.read_csv("../data/auto-mpg.csv")
X = data[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year']].values
y = data[['mpg']].values

scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y)

model1 = SGDRegression()
model1.fit(X, y)
y_pred = model1.predict(X)
# Accuracy
accuracy1 = model1.score(X, y)
print("Accuracy SGDRegression: ",accuracy1)

# # print weights
# print("Weights",model1.weights)
# # print y_pred
# print("y_pred",y_pred)
# pdiff = model1.pdiff(y, y_pred)
# print("Pdiff",pdiff)
# rss = model1.rss(pdiff)
# print("RSS",rss)


# do the same for model2
model2 = ClosedFormRegression()
model2.fit(X, y)
y_pred = model2.predict(X)
# Accuracy
accuracy2 = model2.score(X, y)
print("Accuracy ClosedFormRegression: ",accuracy2)

# print weights
# print("Weights 2:",model2.weights)
# # print y_pred
# print("y_pred 2:",y_pred)
# pdiff = model2.pdiff(y, y_pred)
# print("Pdiff 2:",pdiff)
# rss = model2.rss(pdiff)
# print("RSS 2:",rss)