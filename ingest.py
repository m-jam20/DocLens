"""Build the knowledge base: parse PDFs -> chunk -> embed -> store in Chroma."""

import os
import glob

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DATA_DIR = os.getenv("DATA_DIR", "data")
CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
COLLECTION = "documents"

CHUNK_WORDS = 250
OVERLAP_WORDS = 50


def read_words(pdf_path):
    """Flatten the whole PDF into (word, pdf_page_number) pairs.

    Chunking over this instead of per-page keeps a section that straddles a page
    break in one chunk, while still letting each chunk report its page range.
    """
    reader = PdfReader(pdf_path)
    words = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        for word in text.split():
            words.append((word, page_number))
    return words


def chunk_words(words):
    step = CHUNK_WORDS - OVERLAP_WORDS
    for start in range(0, len(words), step):
        window = words[start : start + CHUNK_WORDS]
        if len(window) > 20:
            text = " ".join(w for w, _ in window)
            yield text, window[0][1], window[-1][1]


def main():
    pdf_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))
    if not pdf_paths:
        raise SystemExit(f"No PDFs found in {DATA_DIR}/")

    print(f"Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    # Rebuild from scratch so re-running never duplicates chunks.
    if any(c.name == COLLECTION for c in client.list_collections()):
        client.delete_collection(COLLECTION)
    collection = client.create_collection(COLLECTION)

    ids, documents, metadatas = [], [], []
    for pdf_path in pdf_paths:
        source = os.path.basename(pdf_path)
        count = 0
        for i, (text, page_start, page_end) in enumerate(chunk_words(read_words(pdf_path))):
            ids.append(f"{source}:c{i}")
            documents.append(text)
            metadatas.append(
                {"source": source, "page_start": page_start, "page_end": page_end}
            )
            count += 1
        print(f"  {source}: {count} chunks")

    print(f"Embedding {len(documents)} chunks...")
    embeddings = embedder.encode(documents, batch_size=64, show_progress_bar=True)

    print("Storing in Chroma...")
    for start in range(0, len(documents), 500):
        end = start + 500
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end].tolist(),
            metadatas=metadatas[start:end],
        )

    print(f"Done. Collection '{COLLECTION}' now holds {collection.count()} chunks.")


if __name__ == "__main__":
    main()
