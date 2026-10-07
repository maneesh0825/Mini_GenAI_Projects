"""
PROJECT TITLE: Multi-Head Attention Shape Demonstration
CONCEPT: Multi-Head Attention
"""
import numpy as np

sentence = "The dog likes food"
tokens = sentence.split()

embedding_dim = 8

X = np.random.rand(len(tokens), embedding_dim)

for num_heads in [1, 2, 4]:
    if embedding_dim % num_heads != 0:
        continue

    head_dim = embedding_dim // num_heads
    reshaped = X.reshape(len(tokens), num_heads, head_dim)

    print("\nNumber of heads:", num_heads)
    print("Input shape:", X.shape)
    print("Head shape:", reshaped.shape)
