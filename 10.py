# pip install langchain langchain-community langchain-cohere pypdf faiss-cpu sentence-transformers

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import ChatCohere

# API Key
os.environ["COHERE_API_KEY"] = "YOUR_API_KEY"

# Load PDF
loader = PyPDFLoader("IPC.pdf")
docs = loader.load()

# Split text
splitter = RecursiveCharacterTextSplitter(chunk_size=500)
chunks = splitter.split_documents(docs)

# Embeddings + Vector DB
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(chunks, embeddings)

# Retriever
retriever = db.as_retriever()

# LLM
llm = ChatCohere(model="command-a-03-2025")

# Chat loop
while True:
    q = input("Ask: ")

    if q == "exit":
        break

    docs = retriever.invoke(q)
    context = "\n".join([d.page_content for d in docs])

    prompt = f"Context:\n{context}\n\nQuestion:{q}"

    response = llm.invoke(prompt)

    print("\nAnswer:")
    print(response.content)