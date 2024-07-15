from models.logistic_regression.logistic import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from StandardScaler.StandardScaler import StandardScaler

import numpy as np

# Generate a synthetic dataset
X, y = make_classification(n_samples=10000, n_features=10, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = model.score(X_test, y_test)
print(f'My model accuracy: {accuracy}')


from sklearn.linear_model import LogisticRegression

model2 = LogisticRegression()
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
accuracy2 = model2.score(X_test, y_test)
print(f'Accuracy Sk-learn: {accuracy2}')