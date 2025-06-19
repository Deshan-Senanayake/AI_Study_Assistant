from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

def summarize_whole_pdf(llm, docs):
    combined_text = "\n\n".join(doc.page_content for doc in docs[:6])  # Limit to first 6 chunks
    full_doc = Document(page_content=combined_text)
    chain = load_summarize_chain(llm, chain_type="stuff")  # 'stuff' is faster
    return chain.run([full_doc])
