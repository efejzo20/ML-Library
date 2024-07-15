from models.decision_tree.tree import DecisionTree
from models.ensemble.ensemble import RandomForest, GradientBoostingClassifier
#from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score


# # Load the Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Hyperparameter tuning for Decision Tree
dt_params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
dt = DecisionTree()

dt_grid_search = GridSearchCV(estimator=dt, param_grid=dt_params, cv=5, scoring='accuracy')
dt_grid_search.fit(X_train, y_train)

best_dt = dt_grid_search.best_estimator_
train_accuracy_dt = best_dt.accuracy(X_train, y_train)
test_accuracy_dt = best_dt.accuracy(X_test, y_test)
mean_cv_accuracy_dt = dt_grid_search.best_score_

print(f"Decision Tree Training accuracy: {train_accuracy_dt:.2f}")
print(f"Decision Tree Testing accuracy: {test_accuracy_dt:.2f}")
print(f"Decision Tree Mean accuracy (k-fold): {mean_cv_accuracy_dt:.2f}")

# Hyperparameter tuning for Random Forest
rf_params = {
    'num_trees': [10, 25, 50],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10]
}
rf = RandomForest()
rf_grid_search = GridSearchCV(estimator=rf, param_grid=rf_params, cv=5, scoring='accuracy')
rf_grid_search.fit(X_train, y_train)

best_rf = rf_grid_search.best_estimator_
train_accuracy_rf = best_rf.accuracy(X_train, y_train)
test_accuracy_rf = best_rf.accuracy(X_test, y_test)
mean_cv_accuracy_rf = rf_grid_search.best_score_

print(f"Random Forest Training accuracy: {train_accuracy_rf:.2f}")
print(f"Random Forest Testing accuracy: {test_accuracy_rf:.2f}")
print(f"Random Forest Mean accuracy (k-fold): {mean_cv_accuracy_rf:.2f}")



# ---------------- Random Forest-----------------------

# # Initialize the RandomForest classifier
# rf = RandomForest(num_trees=25, min_samples_split=2, max_depth=5)

# # Train the RandomForest classifier
# rf.fit(X_train, y_train)

# # Calculate accuracy on the training set
# train_accuracy = rf.accuracy(X_train, y_train)
# print(f"Training accuracy: {train_accuracy:.2f}")

# # Calculate accuracy on the testing set
# test_accuracy = rf.accuracy(X_test, y_test)
# print(f"Testing accuracy: {test_accuracy:.2f}")

# # Perform k-fold cross-validation
# mean_cv_accuracy, std_cv_accuracy = rf.cross_validate(X, y, k=5)
# print(f"Mean accuracy (k-fold): {mean_cv_accuracy:.2f}")
# print(f"Standard deviation of accuracy (k-fold): {std_cv_accuracy:.2f}")



