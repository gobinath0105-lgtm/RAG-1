
from sentence_transformers import SentenceTransformer
import numpy as np


def create_embeddings(all_chunks):

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-V2")

    # Extract text from chunks
    texts = [chunk["context"] for chunk in all_chunks]

    # Generate embeddings
    embeddings = model.encode(texts)

    # Convert to float32 for FAISS
    embeddings = np.array(embeddings, dtype="float32")

    return model, texts, embeddings
