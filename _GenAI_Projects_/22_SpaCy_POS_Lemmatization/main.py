"""
PROJECT TITLE: POS Tagging and Lemmatization using spaCy
CONCEPTS: Part-of-Speech Tagging, Lemmatization
"""
import spacy

nlp = spacy.load("en_core_web_sm")

text = "Elon Musk founded SpaceX in the United States."
doc = nlp(text)

for token in doc:
    print(token.text, "->", token.pos_, "->", token.lemma_)
