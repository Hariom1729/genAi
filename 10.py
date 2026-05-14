# pip install langchain langchain-community langchain-cohere pypdf faiss-cpu sentence-transformers

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_cohere import ChatCohere

# API Key
os.environ["COHERE_API_KEY"] = "YOUR_API_KEY"

# Load PDF
docs = PyPDFLoader("IPC.pdf").load()

# Create vector database
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embeddings)

# Retriever + Model
retriever = db.as_retriever()
llm = ChatCohere(model="command-a-03-2025")

# Chat loop
while True:
    q = input("Ask: ")

    if q == "exit":
        break

    # Get relevant context
    context = retriever.invoke(q)[0].page_content

    # Generate answer
    response = llm.invoke(f"Context:\n{context}\n\nQuestion: {q}")

    print("\nAnswer:")
    print(response.content)