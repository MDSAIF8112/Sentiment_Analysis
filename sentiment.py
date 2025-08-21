# Sentiment Analysis Basic Project
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Sample dataset (you can add more sentences)
texts = [
    "I love this product, it is amazing!",
    "This is the worst thing I bought.",
    "I am happy with the service.",
    "I am very disappointed and sad.",
    "It is okay, not that great."
]
labels = ["positive", "negative", "positive", "negative", "neutral"]

# Step 1: Convert text to numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Step 2: Train the model
model = LogisticRegression()
model.fit(X, labels)

# Step 3: Take user input
user_input = input("Enter a message: ")

# Step 4: Transform input and predict
user_vec = vectorizer.transform([user_input])
prediction = model.predict(user_vec)

print(f"Sentiment: {prediction[0]}")
