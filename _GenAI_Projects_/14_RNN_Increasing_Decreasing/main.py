"""
PROJECT TITLE: Increasing vs Decreasing Sequence Classification
ALGORITHM: Simple RNN
"""
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

X = np.array([
    [[1], [2], [3], [4]],
    [[2], [3], [4], [5]],
    [[5], [4], [3], [2]],
    [[8], [7], [6], [5]],
    [[1], [3], [5], [7]],
    [[7], [5], [3], [1]]
], dtype=float)

y = np.array([1, 1, 0, 0, 1, 0])

model = Sequential([
    SimpleRNN(16, activation="tanh", input_shape=(4, 1)),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=200, verbose=0)

test = np.array([[[2], [4], [6], [8]]], dtype=float)
print("Increasing probability:", model.predict(test, verbose=0)[0][0])
