"""
PROJECT TITLE: Bag of Words (BoW) Text Representation
CONCEPT: CountVectorizer
"""
from sklearn.feature_extraction.text import CountVectorizer

sentences = [
    "I love Python",
    "I love machine learning",
    "Python is powerful"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)

print("Vocabulary:", vectorizer.get_feature_names_out())
print("BoW Matrix:\n", X.toarray())
