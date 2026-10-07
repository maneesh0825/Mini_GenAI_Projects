"""
PROJECT TITLE: Self-Attention from Scratch using NumPy
CONCEPT: Q @ K.T -> Scaling -> Softmax -> Attention @ V
"""
import numpy as np

X = np.array([
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0]
])

WQ = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
])

WK = WQ.copy()
WV = WQ.copy()

Q = X @ WQ
K = X @ WK
V = X @ WV

scores = Q @ K.T
scaled_scores = scores / np.sqrt(K.shape[1])

exp_scores = np.exp(scaled_scores - np.max(scaled_scores, axis=1, keepdims=True))
attention_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

output = attention_weights @ V

print("Q:\n", Q)
print("K:\n", K)
print("V:\n", V)
print("Attention Weights:\n", attention_weights)
print("Attention Output:\n", output)
