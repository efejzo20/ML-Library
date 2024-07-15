from models.DL.mlp import CNNClassifier, ConvLayer, ReLULayer, PoolingLayer
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


rng = np.random.RandomState(42)
layers = [
    ConvLayer(in_channels=1, out_channels=16, kernel_size=(3, 3)),
    ReLULayer(),
    #PoolingLayer(kernel_size=(2, 2)),
    ConvLayer(in_channels=16, out_channels=16, kernel_size=(3, 3)),
    ReLULayer(),
    #PoolingLayer(kernel_size=(2, 2))
]
cnn = CNNClassifier(layers, random_state=42, input_shape=(8, 8, 1))

data = load_digits()
X, y = np.expand_dims(data['images'], axis=-1), data['target']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42, shuffle=True)
cnn.fit(X_train, y_train)

print('accuracy', cnn.score(X_test, y_test))