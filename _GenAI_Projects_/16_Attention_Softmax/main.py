"""
PROJECT TITLE: Attention Softmax Weight Calculation
CONCEPT: Attention weights using Softmax
"""
import numpy as np

scores = np.array([2.0, 1.0, 0.5])

exp_scores = np.exp(scores - np.max(scores))
attention_weights = exp_scores / np.sum(exp_scores)

print("Scores:", scores)
print("Attention weights:", attention_weights)
print("Sum of weights:", np.sum(attention_weights))
