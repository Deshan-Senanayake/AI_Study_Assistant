from langchain.chains import RetrievalQA

def create_qa_chain(llm, vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", k=3)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain
