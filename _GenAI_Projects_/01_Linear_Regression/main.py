"""
PROJECT TITLE: Student Score Prediction using Linear Regression
ALGORITHM: Linear Regression
"""
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3], [4], [5]])       # Hours studied
y = np.array([35, 45, 50, 60, 70])            # Scores

model = LinearRegression()
model.fit(X, y)

hours = np.array([[6]])
prediction = model.predict(hours)

print("Predicted score for 6 hours:", prediction[0])
