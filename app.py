
import streamlit as st

from sentence_transformers import SentenceTransformer
from vector_database import load_vector_database
from storage import load_chunks
from retriver import retrieve_chunks
from generator import generate_answer


# =====================================
# 1. PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="My RAG Assistant",
    page_icon="🤖",
    layout="centered"
)
# SIDEBAR



# =====================================
# 2. LOAD RAG RESOURCES
# =====================================

@st.cache_resource
def load_rag_resources():

    model = SentenceTransformer("all-MiniLM-L6-V2")

    index = load_vector_database(
        path="database/faiss.index"
    )

    chunks = load_chunks(
        path="database/chunks.json"
    )

    return model, index, chunks


with st.spinner("Loading AI Assistant..."):

    model, index, chunks = load_rag_resources()

with st.sidebar:

    st.title("🤖 RAG Assistant")

    st.write("PDF-Based Question Answering")

    st.divider()

    st.subheader("Project Information")

    st.write("📄 Document: Resume PDF")
    st.write("🧠 Embedding: MiniLM-L6-V2")
    st.write("🗄️ Database: FAISS")
    st.write("⚡ LLM: Groq")

    st.divider()

    st.write(f"Total Chunks: {len(chunks)}")

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()
# =====================================
# 3. APPLICATION HEADER
# =====================================

st.title("🤖 My RAG Assistant")

st.caption(
    "Ask questions about my resume using AI."
)

st.success("RAG database connected successfully!")


# =====================================
# 4. CHAT HISTORY
# =====================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# Display previous messages

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# =====================================
# 5. USER INPUT
# =====================================

query = st.chat_input(
    "Ask something about my resume..."
)


# =====================================
# 6. RAG QUESTION ANSWERING
# =====================================

if query:

    # Store user message

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):

        st.markdown(query)


    # Generate AI answer

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                # Retrieve relevant chunks

                results = retrieve_chunks(
                    query=query,
                    model=model,
                    index=index,
                    all_chunks=chunks,
                    top_k=5
                )


                # Generate answer using Groq

                answer = generate_answer(
                    query=query,
                    results=results
                )


                # Display AI answer

                st.markdown(answer)


                # Display retrieved sources

                with st.expander("📚 View Retrieved Sources"):

                    for i, result in enumerate(results, start=1):

                        st.markdown(f"### Source {i}")

                        st.write(
                            "Page:",
                            result["metadata"]["page"]
                        )

                        st.write(
                            "Similarity:",
                            round(result["similarity"], 3)
                        )

                        st.write(
                            "Context:",
                            result["context"]
                        )

                        st.divider()


                # Store assistant response

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })


            except Exception as e:

                st.error(f"Error: {e}")
