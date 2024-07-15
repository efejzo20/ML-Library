from models.naive_bayes.naive_bayes import MultinomialNaiveBayes

import numpy as np


# Sample training data (x: documents, y: labels)
x_train = np.array([
    ['chinese', 'beijing', 'chinese'],
    ['chinese', 'chinese', 'shanghai'],
    ['chinese', 'macao'],
    ['tokyo', 'japan', 'chinese']
])

y_train = np.array([0, 0, 0, 1])  # Labels (0: Chinese text, 1: Japanese text)

# Create an instance of the MultinomialNaiveBayes
model = MultinomialNaiveBayes(alpha=1.0)

# Train the model
model.fit(x_train, y_train)

# Sample test data
x_test = np.array([
    ['chinese', 'chinese', 'chinese', 'tokyo', 'japan'],
    ['tokyo', 'japan', 'japan']
])

# Make predictions
predictions = model.predict(x_test)
print("Predictions:", predictions)

# Predict probabilities
probabilities = model.predict_proba(x_test)
print("Prediction Probabilities:", probabilities)