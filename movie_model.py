import random
import re
import pandas as pd
import nltk
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier

# Importing the training data
imdb_data = pd.read_csv('IMDB Dataset.csv')
train_reviews = imdb_data['review'][:40000]
train_sentiments = imdb_data['sentiment'][:40000]
test_reviews = imdb_data['review'][40000:]
test_sentiments = imdb_data['sentiment'][40000:]

# Preprocessing functions
stopwords = set(stopwords.words('english'))

def preprocess_text(text):
    # Remove square brackets and contents within them
    text = re.sub('\[[^]]*\]', '', text)

    # Tokenize the text
    tokens = word_tokenize(text.lower())

    # Remove stopwords
    tokens = [token for token in tokens if token.lower() not in stopwords]

    return tokens

# Preprocess the training and test reviews
train_features = [(preprocess_text(review), sentiment) for review, sentiment in zip(train_reviews, train_sentiments)]
test_features = [(preprocess_text(review), sentiment) for review, sentiment in zip(test_reviews, test_sentiments)]

# Define the feature extraction function
def extract_features(words):
    return dict([(word, True) for word in words])

# Extract features from the training and test data
train_set = [(extract_features(review), sentiment) for review, sentiment in train_features]
test_set = [(extract_features(review), sentiment) for review, sentiment in test_features]

# Train the NaiveBayes classifier
classifier = NaiveBayesClassifier.train(train_set)

# Evaluate the model on the test data
predictions = [classifier.classify(review) for review, _ in test_set]
accuracy = accuracy_score(test_sentiments, predictions)

# testing a review in the model
features, sentiment = test_set[10]
prediction = classifier.classify(features)
print("Review:", test_reviews.iloc[10])
print("Predicted Sentiment:", prediction)
print("Actual Sentiment:", sentiment)
print("-----------------------")

print("Accuracy:", accuracy)