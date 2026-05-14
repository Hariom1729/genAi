# pip install gensim matplotlib scikit-learn

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
import numpy as np

# Sample corpus
corpus = [
    "patient diagnosed with diabetes",
    "treatment includes antibiotics",
    "vaccine prevents infections",
    "doctor recommends therapy"
]

# Preprocess
data = [s.lower().split() for s in corpus]

# Train model
model = Word2Vec(data, vector_size=50, min_count=1)

# Get words and vectors
words = list(model.wv.index_to_key)
vectors = np.array([model.wv[w] for w in words])

# t-SNE
tsne = TSNE(n_components=2, perplexity=3, random_state=42)
points = tsne.fit_transform(vectors)

# Plot
for i, word in enumerate(words):
    plt.scatter(points[i, 0], points[i, 1])
    plt.text(points[i, 0], points[i, 1], word)

plt.title("Medical Word Embeddings")
plt.show()

# Similar words
print(model.wv.most_similar("treatment", topn=3))