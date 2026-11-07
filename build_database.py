from document_processor import process_pdf
from Embedding import create_embeddings
from vector_database import (
    create_vector_database,
    save_vector_database
)
from storage import save_chunks

# 1. Process PDF
pdf_path = "data/Gobinath_Resume.pdf"
all_chunks = process_pdf(pdf_path)

print("Total chunks:", len(all_chunks))

# 2. Generate embeddings
model, texts, embeddings = create_embeddings(all_chunks)

print("Embedding shape:", embeddings.shape)

# 3. Create FAISS database
index = create_vector_database(embeddings)

# 4. Save FAISS database
save_vector_database(index, path="database/faiss.index")

# 5. Save chunks
save_chunks(all_chunks, path="database/chunks.json")

print("DATABASE BUILD COMPLETED!")