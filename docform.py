import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).parent
PDF = ROOT / "data" / "cnv1.pdf"
DB = ROOT / "chroma_db"
MODEL = "openai/gpt-oss-120b"

PROMPT = """Answer the question using only the context below.
If you don't know the answer, just say "I don't know".

Context:
{context}

Question: {query}

Answer:"""


def build_store() -> Chroma:
    if not PDF.exists():
        raise FileNotFoundError(f"No PDF found at {PDF}")

    store = Chroma(
        collection_name="vectorDB",
        embedding_function=HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        ),
        persist_directory=str(DB),
    )

    if not store.get()["ids"]:
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150
        ).split_documents(PyPDFLoader(str(PDF)).load())
        store.add_documents(chunks)
        print(f"Indexed {len(chunks)} chunks from {PDF.name}")

    return store


def ask(query: str) -> str:
    docs = build_store().as_retriever(search_kwargs={"k": 4}).invoke(query)
    context = "\n\n".join(d.page_content for d in docs)

    llm = ChatGroq(model=MODEL, temperature=0)
    return llm.invoke(PROMPT.format(context=context, query=query)).content


if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    question = " ".join(sys.argv[1:]) or "Explain strong AI"
    print(ask(question))