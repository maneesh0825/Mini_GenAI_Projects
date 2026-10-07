"""
PROJECT TITLE: TF-IDF Text Representation
CONCEPT: Term Frequency-Inverse Document Frequency
"""
from sklearn.feature_extraction.text import TfidfVectorizer

sentences = [
    "the cat drinks milk",
    "the dog drinks milk",
    "the cat likes food"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences)

print("Vocabulary:", vectorizer.get_feature_names_out())
print("TF-IDF Matrix:\n", X.toarray())
