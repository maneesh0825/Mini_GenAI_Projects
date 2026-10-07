"""
PROJECT TITLE: Word Embedding Comparison
CONCEPTS: Word2Vec, FastText, GloVe

This file demonstrates the idea of word embeddings.
Word2Vec and FastText can be trained with gensim.
GloVe is commonly loaded from pretrained vectors.
"""
from gensim.models import Word2Vec, FastText

sentences = [
    ["i", "love", "python"],
    ["python", "is", "powerful"],
    ["i", "love", "machine", "learning"]
]

word2vec = Word2Vec(sentences, vector_size=50, window=3, min_count=1, workers=1)
fasttext = FastText(sentences, vector_size=50, window=3, min_count=1, workers=1)

print("Word2Vec vector for python:")
print(word2vec.wv["python"][:10])

print("\nFastText vector for python:")
print(fasttext.wv["python"][:10])

print("\nGloVe: use pretrained GloVe vectors when available.")
