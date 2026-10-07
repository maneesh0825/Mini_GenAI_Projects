"""
PROJECT TITLE: Linear Regression using Gradient Descent
ALGORITHM: Gradient Descent
"""
import numpy as np

X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 6, 8, 10], dtype=float)

w = 0.0
b = 0.0
learning_rate = 0.01

for epoch in range(1000):
    y_pred = w * X + b

    dw = np.mean((y_pred - y) * X)
    db = np.mean(y_pred - y)

    w -= learning_rate * dw
    b -= learning_rate * db

print("Weight:", w)
print("Bias:", b)
print("Prediction for x=6:", w * 6 + b)
