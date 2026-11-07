from generator import generate_answer
from retriver import retrieve_chunks
from vector_database import load_vector_database
from storage import load_chunks
from Embedding import create_embeddings

print("LOADING RAG DATABASE...")

# Load existing chunks
loaded_chunks = load_chunks(path="database/chunks.json")

# Load existing FAISS database
index = load_vector_database(path="database/faiss.index")

# Load embedding model
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-V2")

print("DATABASE LOADED SUCCESSFULLY!")
print("Total chunks:", len(loaded_chunks))
print("Total vectors:", index.ntotal)

print("\n========== RAG CHATBOT ==========")

while True:

    query = input("\nAsk a question (type exit to stop): ")

    if query.lower() == "exit":
        print("Chatbot closed!")
        break

    results = retrieve_chunks(
        query=query,
        model=model,
        index=index,
        all_chunks=loaded_chunks,
        top_k=5
    )

    answer = generate_answer(
        query=query,
        results=results
    )

    print("\nAI Answer:")
    print(answer)