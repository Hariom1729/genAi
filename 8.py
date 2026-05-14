# pip install langchain langchain-community langchain-cohere cohere

import os
from langchain_community.document_loaders import TextLoader
from langchain_cohere import ChatCohere

# API Key
os.environ["COHERE_API_KEY"] = "YOUR_API_KEY"

# Load file
loader = TextLoader("teaching.txt")
docs = loader.load()

text = docs[0].page_content

# Load model
llm = ChatCohere(model="command-a-03-2025")

# Chat loop
while True:
    q = input("Ask: ")

    if q == "exit":
        break

    prompt = f"Document:\n{text}\n\nQuestion: {q}"

    response = llm.invoke(prompt)

    print("\nAnswer:")
    print(response.content)