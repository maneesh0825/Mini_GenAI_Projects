"""
PROJECT TITLE: Sine Wave Next-Value Prediction using LSTM
ALGORITHM: LSTM
"""
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense

t = np.linspace(0, 20, 300)
values = np.sin(t)

sequence_length = 10
X, y = [], []

for i in range(len(values) - sequence_length):
    X.append(values[i:i + sequence_length])
    y.append(values[i + sequence_length])

X = np.array(X).reshape(-1, sequence_length, 1)
y = np.array(y)

model = Sequential([
    LSTM(32, input_shape=(sequence_length, 1)),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=20, verbose=0)

test = X[-1:]
print("Predicted next sine value:", model.predict(test, verbose=0)[0][0])
