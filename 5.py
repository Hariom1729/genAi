# pip install gensim

import gensim.downloader as api
import random

# Load model
model = api.load("glove-wiki-gigaword-100")

# Input word
seed = "freedom"

# Get similar words
words = [w for w, _ in model.most_similar(seed, topn=5)]
random.shuffle(words)

# Generate paragraph
print(f"In a world of {seed}, people valued {', '.join(words)}.")