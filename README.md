# pdf-reader

Ask questions about a local PDF. The document is chunked and embedded into a local
vector store, the most relevant passages are retrieved for each question, and Groq's
Llama-family model answers using only that retrieved text.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a free
[Groq API key](https://console.groq.com/keys).

```bash
uv sync
```

Create a `.env` in the project root:

```
GROQ_API_KEY=gsk_...
```

Then drop a PDF at `data/cnv1.pdf` (the path is set in `docform.py`).

## Usage

```bash
uv run docform.py "What is strong AI?"
uv run docform.py                    # falls back to a default question
```

Example output:

```
Strong AI, also called General AI, is an artificial intelligence that aims to mimic
human intelligence and perform a wide range of tasks...
```

## How it works

1. `PyPDFLoader` reads the PDF into pages.
2. `RecursiveCharacterTextSplitter` splits them into 1000-character chunks with a
   150-character overlap, so sentences are not cut mid-thought.
3. `HuggingFaceEmbeddings` (`all-mpnet-base-v2`) turns each chunk into a vector and
   `Chroma` stores them in `chroma_db/`.
4. Each question retrieves the 4 nearest chunks and passes them to `ChatGroq` inside a
   prompt that forbids using outside knowledge.

Indexing only happens once. If `chroma_db/` is already populated, subsequent runs
reuse it, so repeat runs stay fast and do not accumulate duplicate chunks.

## Configuration

Edit the constants at the top of `docform.py`:

| Constant  | Purpose                                            |
| --------- | -------------------------------------------------- |
| `PDF`     | Path to the PDF you want to read                    |
| `DB`      | Where the vector index is written                   |
| `MODEL`   | Groq model id                                       |

Groq retires models fairly often. To see the ones your key can reach:

```bash
uv run python -c "import os; from dotenv import load_dotenv; from groq import Groq; load_dotenv(); print(*sorted(m.id for m in Groq(api_key=os.getenv('GROQ_API_KEY')).models.list().data), sep='\n')"
```

## Notes

- `langchain-community` prints a deprecation warning on import. `PyPDFLoader` is being
  split out into a standalone package; migrating to `langchain-pypdf` would silence it.
- Set `HF_TOKEN` in `.env` to raise Hugging Face Hub rate limits. It is only needed on
  the first run, while the embedding model downloads.
- Delete `chroma_db/` to force a rebuild after changing `PDF` or the chunk settings.

## Layout

```
docform.py        entry point: build the index, retrieve, answer
data/cnv1.pdf     your source document (gitignored)
chroma_db/        vector index (gitignored, regenerated on demand)
```