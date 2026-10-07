"""
PROJECT TITLE: Next Value Prediction using RNN
ALGORITHM: Simple RNN
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

X = np.array([
    [[1], [2], [3]],
    [[2], [3], [4]],
    [[3], [4], [5]],
    [[4], [5], [6]]
], dtype=float)

y = np.array([4, 5, 6, 7], dtype=float)

model = Sequential([
    SimpleRNN(32, activation="tanh", input_shape=(3, 1)),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=200, verbose=0)

test = np.array([[[5], [6], [7]]], dtype=float)
print("Predicted next value:", model.predict(test, verbose=0)[0][0])
