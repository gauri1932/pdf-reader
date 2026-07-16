from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
# from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_groq import ChatGroq

data = PyPDFLoader("./data/cnv1.pdf")
document=data.load()
# print((document[200]))
# print(len(document))

texts_splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200    # we need over lap because a sentence might cut in between two chunks resulting in not proper chunks
)

chunks=texts_splitter.split_documents(document)
# print(len(chunks))

# texts = [doc.page_content for doc in chunks]
embedding= HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# embeddings = embedding.embed_documents(texts)

# model = SentenceTransformer(
#     "sentence-transformers/all-mpnet-base-v2"
# )

# embeddings = model.encode(texts)
# print(len(embeddings))     
# If you're learning how embeddings work	Use
# Understand the embedding model itself	SentenceTransformer(...).encode()
# Build a LangChain RAG pipeline	HuggingFaceEmbeddings(...).embed_documents()

# print(embeddings[0])

vector_store = Chroma(
    collection_name="vectorDB",
    embedding_function=embedding,
    persist_directory="./chroma_db"
)
vector_store.add_documents(chunks)    # embedd the document internally no need for finding embeddings alag se

retriever=vector_store.as_retriever(
    search_kwargs={"k":2}
)
query="What is Strong AI?"

docs=retriever.invoke(query)
# print(docs)
# print(type(docs))

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

context="\n\n".join(doc.page_content for doc in docs)

prompt=f"""
Answer the question using only the context below

Context:
{context}

Question:
{query}

Answer:
"""

# response=llm.invoke(prompt)
# print(response.content)

import os
from dotenv import load_dotenv

load_dotenv()

print(os.getenv("GROQ_API_KEY") is not None)