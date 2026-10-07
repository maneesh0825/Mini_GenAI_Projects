"""
PROJECT TITLE: Iris Dataset Visualization using t-SNE
ALGORITHM: t-SNE
"""
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE

data = load_iris()
X, y = data.data, data.target

X_2d = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X)

plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y)
plt.title("t-SNE Visualization of Iris Dataset")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()
