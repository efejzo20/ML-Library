from models.svm.svm import SVR,SVC
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.metrics import accuracy_score, classification_report


X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)
y = np.where(y == 0, -1, 1)  # Convert labels from {0, 1} to {-1, 1}

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate and train the SVC model
svc = SVC(lr=0.01, max_iter=1000)
svc.fit(X_train, y_train)

# Evaluate the model
accuracy, precision, recall = svc.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# Detailed classification report
y_pred = svc.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Class -1', 'Class 1']))





X, y = make_regression(n_samples=100, n_features=1, noise=0.1)

# Instantiate and train the SVR model
svr = SVR(lr=0.01, max_iter=1000, epsilon=0.1)
svr.fit(X, y)

# Evaluate the model
mse = svr.evaluate(X, y)
print(f"Mean Squared Error: {mse}")