import random
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from nltk.classify import NaiveBayesClassifier


class Preprocessor:
    def __init__(self):
        self.stopwords = set(stopwords.words('english'))
        self.contractions = {
                    "aren't": "are not",
                    "can't": "can not",
                    "couldn't": "could not",
                    "didn't": "did not",
                    "don't": "do not",
                    "doesn't": "does not",
                    "hadn't": "had not",
                    "haven't": "have not",
                    "he's": "he has",
                    "he'll": "he will",
                    "he'd": "he would",
                    "here's": "here is",
                    "I'm": "I am",
                    "I've": "I have",
                    "I'll": "I will",
                    "I'd": "I had",
                    "isn't": "is not",
                    "it's": "it has",
                    "it'll": "it will",
                    "mustn't": "must not",
                    "she's": "she has",
                    "she'll": "she will",
                    "she'd": "she had",
                    "shouldn't": "should not",
                    "that's": "that is",
                    "there's": "there is",
                    "they're": "they are",
                    "they've": "they have",
                    "they'll": "they will",
                    "they'd": "they had",
                    "wasn't": "was not",
                    "we're": "we are",
                    "we've": "we have",
                    "we'll": "we will",
                    "we'd": "we had",
                    "weren't": "were not",
                    "what's": "what is",
                    "where's": "where is",
                    "who's": "who is",
                    "who'll": "who will",
                    "won't": "will not",
                    "wouldn't": "would not",
                    "you're": "you are",
                    "you've": "you have",
                    "you'll": "you will",
                    "you'd": "you had"
                }

    def preprocess_text(self, text):
        # Tokenization and contraction expansion
        tokens = word_tokenize(text)
        tokens = [self.contractions.get(token, token) for token in tokens]

        # Removing stopwords and non-alphanumeric characters
        clean_tokens = [
            token.lower() for token in tokens
            if token.isalnum() and token.lower() not in self.stopwords
        ]

        return clean_tokens


preprocessor = Preprocessor()

def extract_features(text):
    return {word: True for word in text}


reviews = []
with open('IMDB Dataset.csv', 'r') as file:
    for line in file:
        line = line.strip()
        parts = line.split(',')
        if len(parts) != 2:
            continue
        review = parts[0].replace(",", "")  # Remove commas within the review
        sentiment = parts[1]
        preprocessed_review = preprocessor.preprocess_text(review)
        reviews.append((preprocessed_review, sentiment))

train_size = int(0.8 * len(reviews))
train_reviews = reviews[:train_size]
test_reviews = reviews[train_size:]

train_features = [(extract_features(review), sentiment) for (review, sentiment) in train_reviews]
test_features = [(extract_features(review), sentiment) for (review, sentiment) in test_reviews]

classifier = NaiveBayesClassifier.train(train_features)

accuracy = nltk.classify.accuracy(classifier, test_features)
print('Accuracy:', accuracy)