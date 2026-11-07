
import faiss
import numpy as np


def retrieve_chunks(query, model, index, all_chunks, top_k=5):

    # Convert query into embedding
    query_embedding = model.encode([query])

    query_embedding = np.array(
        query_embedding,
        dtype="float32"
    )

    # Normalize query vector
    faiss.normalize_L2(query_embedding)

    # Search FAISS
    k = min(top_k, index.ntotal)

    similarities, indices = index.search(
        query_embedding, k
    )

    # Collect relevant chunks
    results = []

    for i in range(k):

        chunk_index = indices[0][i]

        # Ignore invalid FAISS indices
        if chunk_index == -1:
            continue

        results.append({
            "chunk_id": int(chunk_index),
            "context": all_chunks[chunk_index]["context"],
            "metadata": all_chunks[chunk_index]["metadata"],
            "similarity": float(similarities[0][i])
        })

    return results
