from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def create_vector_store(documents):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(documents, embedding_model)
    return db
