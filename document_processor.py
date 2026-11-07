from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import re
def process_pdf(pdf_path):
    # Read the PDF file
    file = PdfReader(pdf_path)

    # Extract text from each page and create metadata
    document = []
    for num_page, page in enumerate(file.pages, start=1):
        text = page.extract_text(extraction_mode="layout") or ""
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        if cleaned_text:
            document.append({
                "context": cleaned_text,
                "metadata": {
                    "page": num_page,
                    "source": pdf_path.split("\\")[-1]  # Extract filename from path
                }
            })

    # Split the text into chunks
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    for doc in document:
        chunks = splitter.split_text(doc["context"])
        for chunk in chunks:
            all_chunks.append({
                "context": chunk,
                "metadata": doc["metadata"]
            })

    return all_chunks