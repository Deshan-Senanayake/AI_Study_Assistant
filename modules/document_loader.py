from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_chunk_pdf(path):
    loader = PyPDFLoader(path)
    raw_docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    chunks = splitter.split_documents(raw_docs)

    for i, chunk in enumerate(chunks):
        page = chunk.metadata.get("page", "?")
        chunk.metadata["source"] = f"Chunk {i} | Page {page}"
        chunk.metadata["slide"] = f"Slide {page}"  # Assume slide ≈ page
    return chunks
