"""Core RAG logic shared by the CLI (`python query.py "..."`), app.py and eval.py.

Retrieval is two-stage: a fast vector search pulls CANDIDATES chunks from
Chroma, then a cross-encoder reranks them and the best TOP_K are kept. The
reranker score also decides whether the documents are relevant at all; if
not, the question is answered as a general chat message without sources.
"""

import os
import re
import sys

import chromadb
import ollama
from sentence_transformers import CrossEncoder, SentenceTransformer

CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")
COLLECTION = "documents"
TOP_K = int(os.getenv("TOP_K", "6"))
CANDIDATES = int(os.getenv("CANDIDATES", "20"))
# Reranker score (raw logit) the best chunk must reach for a document answer.
# Calibrated: eval questions scored >= -1.0, small talk ("hi", "thanks") <= -5.0.
RELEVANCE_MIN = float(os.getenv("RELEVANCE_MIN", "-3.0"))
# Within a document answer, chunks scoring below this are dropped as noise
# (the best chunk is always kept). Fewer, cleaner sources = less blending.
KEEP_MIN = float(os.getenv("KEEP_MIN", "0.0"))

RAG_SYSTEM = """You are a helpful assistant that answers from the user's documents.

Rules:
- Use only the numbered sources in the user's message. If they do not contain
  the answer, say so plainly instead of guessing.
- Every sentence that states a fact must end with the number of the source it
  came from, like [2]. A sentence without a citation is not allowed.
- Keep sources separate: never combine details from different sources into one
  description, because different sources often describe different things.
- Attribute a property only to the thing the source states it for. If a source
  says "a mesh can have X", do not say something else has X.
- Be concise."""

CHAT_SYSTEM = """You are a friendly, helpful assistant. The user's documents had nothing
relevant to this message, so answer from general knowledge, briefly. Do not
invent citations."""

CITATION = re.compile(r"\[(\d+)\]")
# A run of 3+ bare citations at the very end ("... [1] [2] [3] [4]") cites nothing.
CITATION_DUMP = re.compile(r"(?:\s*\[\d+\]){3,}\s*$")


def cite(meta):
    name = meta["source"].removesuffix(".pdf")
    if meta["page_start"] == meta["page_end"]:
        return f"{name}, p.{meta['page_start']}"
    return f"{name}, p.{meta['page_start']}-{meta['page_end']}"


def load_embedder():
    return SentenceTransformer(EMBED_MODEL)


def load_reranker():
    return CrossEncoder(RERANK_MODEL)


def get_client():
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


def load_collection():
    return get_client().get_or_create_collection(COLLECTION)


def retrieve(embedder, collection, question, k=TOP_K, reranker=None, candidates=CANDIDATES):
    """Return the top-k chunks as dicts with text, meta, distance (and score if reranked)."""
    if collection.count() == 0:
        return []
    n = min(candidates if reranker else k, collection.count())
    results = collection.query(
        query_embeddings=[embedder.encode(question).tolist()], n_results=n
    )
    hits = [
        {"text": text, "meta": meta, "distance": distance}
        for text, meta, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]
    if reranker:
        scores = reranker.predict([(question, h["text"]) for h in hits])
        for h, s in zip(hits, scores):
            h["score"] = float(s)
        hits.sort(key=lambda h: h["score"], reverse=True)
    return hits[:k]


def is_relevant(hits):
    """Do the documents answer this? Needs reranker scores on the hits."""
    return bool(hits) and hits[0].get("score", 0.0) >= RELEVANCE_MIN


def keep_strong(hits):
    """Drop weakly related chunks, always keeping the best one."""
    return hits[:1] + [h for h in hits[1:] if h["score"] >= KEEP_MIN]


def build_prompt(question, hits):
    sources = "\n\n".join(
        f"Source [{i}] ({cite(h['meta'])}):\n{h['text']}" for i, h in enumerate(hits, start=1)
    )
    return (
        f"{sources}\n\nQuestion: {question}\n\n"
        "Answer using only the sources above. End every factual sentence with its [n]."
    )


def build_messages(question, hits, history=()):
    """Chat messages for Ollama. history = earlier {"role", "content"} turns."""
    if hits:
        system, user = RAG_SYSTEM, build_prompt(question, hits)
    else:
        system, user = CHAT_SYSTEM, question
    # Earlier turns give follow-up questions their context; their sources are not resent.
    return [{"role": "system", "content": system}, *history[-6:], {"role": "user", "content": user}]


def clean_answer(answer):
    return CITATION_DUMP.sub("", answer).rstrip()


def citations_used(answer, n_sources):
    """Source numbers cited in the answer that actually exist."""
    return sorted({int(m) for m in CITATION.findall(answer) if 1 <= int(m) <= n_sources})


def generate(messages, model=LLM_MODEL, stream=False):
    """Send chat messages to Ollama. With stream=True, yield the answer in pieces."""
    response = ollama.Client(host=OLLAMA_HOST).chat(
        model=model,
        messages=messages,
        stream=stream,
        options={"temperature": 0.1},
        keep_alive="30m",
    )
    if not stream:
        return response["message"]["content"]
    return (part["message"]["content"] for part in response)


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python query.py "your question"')
    question = " ".join(sys.argv[1:])

    hits = retrieve(load_embedder(), load_collection(), question, reranker=load_reranker())
    if is_relevant(hits):
        hits = keep_strong(hits)
        print(f"\nRetrieved {len(hits)} chunks (reranker score, higher = better):")
        for i, h in enumerate(hits, start=1):
            print(f"  [{i}] {h['score']:.3f}  {cite(h['meta'])}")
    else:
        print("\nNo relevant documents found; answering as general chat.")
        hits = []

    print(f"\nAnswer:\n{clean_answer(generate(build_messages(question, hits)))}")


if __name__ == "__main__":
    main()
