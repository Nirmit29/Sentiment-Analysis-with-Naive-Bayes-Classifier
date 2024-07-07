import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
nltk.download('movie_reviews')
from nltk.corpus import movie_reviews

# Load the IMDb movie reviews dataset
reviews = []
for fileid in movie_reviews.fileids():
    tag, filename = fileid.split('/')
    reviews.append((filename, movie_reviews.raw(fileid), tag))

# Convert to DataFrame
data = pd.DataFrame(reviews, columns=['filename', 'review', 'sentiment'])

# Map 'pos' and 'neg' to 1 and 0
data['sentiment'] = data['sentiment'].map({'pos': 1, 'neg': 0})

# Explore dataset
print(data.head())
print(data['sentiment'].value_counts())

# Feature extraction using CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['review'])
y = data['sentiment']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Naive Bayes classifier
model = MultinomialNB()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Classification report
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
