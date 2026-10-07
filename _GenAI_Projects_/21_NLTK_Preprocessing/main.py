"""
PROJECT TITLE: NLP Text Preprocessing
CONCEPTS: Tokenization, Stemming, Lemmatization
"""
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")

text = "The students are studying and playing games."

tokens = word_tokenize(text)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

print("Tokens:", tokens)

print("\nStemming:")
for word in tokens:
    print(word, "->", stemmer.stem(word))

print("\nLemmatization:")
for word in tokens:
    print(word, "->", lemmatizer.lemmatize(word))
