from models.ANN.mlp import MLPRegressor, MLPClassifier

import numpy as np

X_reg = np.random.rand(100, 5)
y_reg = np.random.rand(100, 1)
reg = MLPRegressor(hidden_layer_sizes=(5, 10), lr=0.01, epochs=1000, random_state=42, activation='relu')
reg.fit(X_reg, y_reg)
predictions_reg = reg.predict(X_reg)
print("Regressor Predictions:", predictions_reg[:5])

# # Classifier example
# X_clf = np.random.rand(100, 5)
# y_clf = np.random.randint(0, 3, 100)
# clf = MLPClassifier(lr=0.01, epochs=1000, random_state=42)
# clf.fit(X_clf, y_clf)
# predictions_clf = clf.predict(X_clf)
# print("Classifier Predictions:", predictions_clf[:5])