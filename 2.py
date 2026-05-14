# pip install gensim matplotlib scikit-learn

import gensim.downloader as api
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load model
model = api.load("word2vec-google-news-300")

# Words
words = ['computer', 'internet', 'software', 'hardware', 'server']
vectors = [model[w] for w in words]

# PCA
pca = PCA(n_components=2)
points = pca.fit_transform(vectors)

# Similar words
print(model.most_similar("computer", topn=5))

# Plot
for i, word in enumerate(words):
    plt.scatter(points[i, 0], points[i, 1])
    plt.text(points[i, 0], points[i, 1], word)

plt.title("Word Embeddings")
plt.show()