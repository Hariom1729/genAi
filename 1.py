# pip install gensim scipy

import gensim.downloader as api
from scipy.spatial.distance import cosine

# Load model
model = api.load("word2vec-google-news-300")

# Similar words
print(model.most_similar("king", topn=5))

# Analogy: king - man + woman = ?
print(model.most_similar(
    positive=["king", "woman"],
    negative=["man"],
    topn=1
))

# Cosine similarity
sim = 1 - cosine(model["king"], model["queen"])
print("Similarity:", round(sim, 4))