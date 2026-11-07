
import faiss
import numpy as np


def create_vector_database(embeddings):

    # Make a copy before normalization
    vectors = np.array(embeddings, dtype="float32").copy()

    # Normalize vectors for cosine similarity
    faiss.normalize_L2(vectors)

    # Get embedding dimensions
    dimension = vectors.shape[1]

    # Create FAISS index
    index = faiss.IndexFlatIP(dimension)

    # Store vectors
    index.add(vectors)

    return index


def save_vector_database(index, path="database/faiss.index"):

    faiss.write_index(index, path)
    print("FAISS database saved successfully!")


def load_vector_database(path="database/faiss.index"):

    index = faiss.read_index(path)
    print("FAISS database loaded successfully!")

    return index
