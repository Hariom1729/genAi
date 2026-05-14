# pip install transformers torch

from transformers import pipeline

# Load model
sentiment = pipeline("sentiment-analysis")

# Input text
texts = [
    "This phone is amazing!",
    "Worst product ever."
]

# Analyze sentiment
results = sentiment(texts)

# Print results
for text, result in zip(texts, results):
    print(text)
    print(result)