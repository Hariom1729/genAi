# pip install sentence-transformers langchain-cohere cohere

import os
from sentence_transformers import SentenceTransformer
from langchain_cohere import ChatCohere

# API Key
os.environ["COHERE_API_KEY"] = "YOUR_API_KEY"

# Models
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

llm = ChatCohere(
    model="command-a-03-2025",
    temperature=0.3
)

# User input
prompt = input("Enter prompt: ")

# Generate embeddings
embedding = embed_model.encode(prompt)

# LLM response
response = llm.invoke(prompt)

# Output
print("\nEmbedding Vector (first 10 values):")
print(embedding[:10])

print("\nLLM Response:")
print(response.content)