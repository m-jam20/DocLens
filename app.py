"""Streamlit chat UI for DocLens.

Messages the documents can answer get a cited answer with its sources; anything
else (greetings, general questions) is answered as a normal chat reply.
Run inside the stack:  docker compose up -d  ->  http://localhost:8501
"""

import itertools

import streamlit as st

import query

st.set_page_config(page_title="DocLens", page_icon="🔎", layout="centered")


@st.cache_resource(show_spinner="Loading models...")
def models():
    return query.load_embedder(), query.load_reranker()


@st.cache_resource
def collection():
    return query.load_collection()


def show_sources(hits, cited):
    label = f"Sources ({len(hits)})" + (f" · cited {', '.join(map(str, cited))}" if cited else "")
    with st.expander(label):
        for i, h in enumerate(hits, start=1):
            mark = "**" if i in cited else ""
            st.markdown(f"{mark}[{i}] {query.cite(h['meta'])}{mark} · score {h['score']:.2f}")
            st.caption(h["text"][:600] + ("…" if len(h["text"]) > 600 else ""))


with st.sidebar:
    st.header("DocLens")
    k = st.slider("Sources per answer", 1, 12, query.TOP_K)
    if st.button("New chat", use_container_width=True):
        st.session_state.messages = []
    st.divider()
    try:
        st.caption(f"Index: {collection().count()} chunks")
    except Exception as exc:  # Chroma down
        st.error(f"Vector DB not reachable: {exc}")
        st.stop()
    st.caption(f"LLM: `{query.LLM_MODEL}` · Embeddings: `{query.EMBED_MODEL}`")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("### Ask me anything")
    st.caption(
        "Questions covered by your documents are answered from them, with sources. "
        "Anything else gets a normal chat reply."
    )

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("hits"):
            show_sources(m["hits"], m.get("cited", []))
        elif m["role"] == "assistant":
            st.caption("General answer, not from your documents.")

if question := st.chat_input("Message DocLens"):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.status("Searching documents...", expanded=False) as status:
            embedder, reranker = models()
            hits = query.retrieve(embedder, collection(), question, k=k, reranker=reranker)
            hits = query.keep_strong(hits) if query.is_relevant(hits) else []
            status.update(
                label=f"Found {len(hits)} relevant sources" if hits else "No relevant documents; answering generally",
                state="complete",
            )

        messages = query.build_messages(question, hits, history)
        try:
            # The model may need to load into the GPU first (up to a minute after idling),
            # so show a placeholder until the first token arrives.
            placeholder = st.empty()
            placeholder.caption("Thinking… (the first message can take up to a minute while the model loads)")
            stream = query.generate(messages, stream=True)
            first = next(stream, "")
            placeholder.empty()
            answer = st.write_stream(itertools.chain([first], stream))
        except Exception as exc:
            st.error(f"LLM call failed: {exc}")
            st.info(f"Is the model pulled? `docker compose exec ollama ollama pull {query.LLM_MODEL}`")
            st.stop()

        cleaned = query.clean_answer(answer)
        if cleaned != answer:  # model appended a bare list of all sources
            answer = cleaned
            st.caption("(Removed a trailing list of uncited source numbers.)")
        cited = query.citations_used(answer, len(hits)) if hits else []
        if hits:
            show_sources(hits, cited)
            if not cited:
                st.caption("⚠️ The model gave no source numbers. Check the answer against the sources.")
        else:
            st.caption("General answer, not from your documents.")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "hits": hits, "cited": cited}
    )
