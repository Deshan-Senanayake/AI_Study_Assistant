from modules.document_loader import load_and_chunk_pdf
from modules.vector_store import create_vector_store
from modules.llm_interface import load_llm
from modules.rag_pipeline import create_qa_chain
from modules.slide_mapper import find_matching_slide
from modules.summarizer import summarize_whole_pdf
from modules.flashcards import generate_flashcards

import gradio as gr

docs, db, chain, llm = None, None, None, None

def load_pdf_and_prepare(path):
    global docs, db, chain, llm
    docs = load_and_chunk_pdf(path.name)
    db = create_vector_store(docs)
    llm = load_llm()
    chain = create_qa_chain(llm, db)
    return "PDF loaded and ready!"

def ask_question(query):
    if not chain:
        return "Please upload and load a PDF first."

    # Try fuzzy slide match first
    matched_slide = find_matching_slide(query, docs)
    if matched_slide:
        return llm(matched_slide.page_content + f"\n\nAnswer this: {query}")
    else:
        return chain.run(query)

def summarize_pdf():
    return summarize_whole_pdf(llm, docs)

def flashcard_output():
    return generate_flashcards(llm, docs, num=5)

with gr.Blocks() as demo:
    gr.Markdown("# 🧠 AI Study Assistant")
    file = gr.File(label="Upload PDF")
    status = gr.Textbox(label="Status")
    file_button = gr.Button("Load & Index")
    file_button.click(fn=load_pdf_and_prepare, inputs=file, outputs=status)

    query = gr.Textbox(label="Ask a Question")
    answer = gr.Textbox(label="Answer")
    query.submit(fn=ask_question, inputs=query, outputs=answer)

    with gr.Row():
        summarize_btn = gr.Button("📄 Summarize Full PDF")
        flashcard_btn = gr.Button("🧠 Generate Flashcards")

    summary_out = gr.Textbox(label="Full PDF Summary")
    flashcard_out = gr.Textbox(label="Flashcards")

    summarize_btn.click(fn=summarize_pdf, inputs=None, outputs=summary_out)
    flashcard_btn.click(fn=flashcard_output, inputs=None, outputs=flashcard_out)

demo.launch()
